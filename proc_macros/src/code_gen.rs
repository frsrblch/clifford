use super::types::*;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};

impl Algebra {
    pub fn define(&self) -> TokenStream {
        let types = self.types().map(|ty| ty.define());

        quote! {
            pub use crate::{Dot, Wedge, Commutator, ZeroValue};

            #( #types )*
        }
    }
}

impl Blade {
    pub fn field(&self) -> Ident {
        if self.0.is_empty() {
            Ident::new("scalar", Span::mixed_site())
        } else {
            let mut output = "e".to_string();

            for i in 1..=self.1.dimensions() {
                if self.0.contains(i) {
                    output.push_str(&i.to_string());
                }
            }

            Ident::new(&output, Span::mixed_site())
        }
    }
}

impl Grade {
    pub fn type_ident(&self) -> Ident {
        let str = match self.0 {
            0 => "f64",
            1 => "Vector",
            2 => "Bivector",
            3 => "Trivector",
            4 => "Quadvector",
            5 => "Pentavector",
            6 => "Hexavector",
            _ => unimplemented!("not implemented for grade: {}", self.0),
        };
        Ident::new(str, Span::mixed_site())
    }
}

impl SubAlgebra {
    fn type_ident(&self) -> Ident {
        let str = match self {
            SubAlgebra::Even(_) => "Even",
            SubAlgebra::Odd(_) => "Odd",
        };
        Ident::new(str, Span::mixed_site())
    }
}

impl Type {
    pub fn type_ident(&self) -> TokenStream {
        match self {
            Type::Zero(_) => quote! { crate::Zero },
            Type::Grade(grade) => grade.type_ident().to_token_stream(),
            Type::SubAlgebra(sub) => sub.type_ident().to_token_stream(),
        }
    }

    pub fn define(&self) -> TokenStream {
        if let Type::Zero(_) = self {
            return quote!();
        }

        let ty = self.type_ident();

        let blades = self.blades().map(|b| {
            let f = b.field();
            quote! { pub #f: f64, }
        });

        let new_fn_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: f64 }
        });

        let new_fn_struct_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f, }
        });

        let zero_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: 0., }
        });

        let neg_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: -self.#f, }
        });

        let add_self_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: self.#f + rhs.#f, }
        });

        let sub_self_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: self.#f - rhs.#f, }
        });

        let mul_f64_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: self.#f * rhs, }
        });

        let f64_mul_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: self * rhs.#f, }
        });

        let div_f64_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: self.#f / rhs, }
        });

        let algebra = self.algebra();
        let type_ops = algebra.types().flat_map(|rhs| {
            BinaryOp::iter().map(move |op| ImplBinaryOp {
                lhs: *self,
                rhs,
                op,
            })
        });

        quote! {
            #[derive(Debug, Default, Copy, Clone, PartialEq)]
            pub struct #ty {
                #( #blades )*
            }

            impl #ty {
                pub const fn new(
                    #(#new_fn_fields,)*
                ) -> Self {
                    Self {
                        #(#new_fn_struct_fields)*
                    }
                }
            }

            impl const crate::ZeroValue for #ty {
                fn zero() -> Self {
                    Self {
                        #(#zero_fields)*
                    }
                }
            }

            impl const std::ops::Neg for #ty {
                type Output = #ty;
                fn neg(self) -> #ty {
                    Self {
                        #(#neg_fields)*
                    }
                }
            }

            impl const std::ops::Add for #ty {
                type Output = #ty;
                fn add(self, rhs: Self) -> Self {
                    Self {
                        #( #add_self_fields )*
                    }
                }
            }

            impl const std::ops::Sub for #ty {
                type Output = #ty;
                fn sub(self, rhs: Self) -> Self {
                    Self {
                        #( #sub_self_fields )*
                    }
                }
            }

            impl const std::ops::Mul<f64> for #ty {
                type Output = Self;
                fn mul(self, rhs: f64) -> Self {
                    Self {
                        #(#mul_f64_fields)*
                    }
                }
            }

            impl const std::ops::Mul<#ty> for f64 {
                type Output = #ty;
                fn mul(self, rhs: #ty) -> Self::Output {
                    #ty {
                        #(#f64_mul_fields)*
                    }
                }
            }

            impl const std::ops::Div<f64> for #ty {
                type Output = Self;
                fn div(self, rhs: f64) -> Self {
                    Self {
                        #(#div_f64_fields)*
                    }
                }
            }

            #( #type_ops )*
        }
    }
}

pub struct ImplBinaryOp {
    pub lhs: Type,
    pub rhs: Type,
    pub op: BinaryOp,
}

#[derive(Debug, Copy, Clone)]
pub enum BinaryOp {
    Mul,
    Dot,
    Wedge,
}

impl BinaryOp {
    pub fn iter() -> impl Iterator<Item = Self> + 'static {
        [Self::Mul, Self::Dot, Self::Wedge].into_iter()
    }

    pub fn call(&self, lhs: Blade, rhs: Blade) -> Product {
        match self {
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Dot => lhs.dot(rhs),
            BinaryOp::Wedge => lhs.wedge(rhs),
        }
    }

    pub fn trait_ty(&self) -> TokenStream {
        match self {
            BinaryOp::Mul => quote! { std::ops::Mul },
            BinaryOp::Dot => quote! { crate::Dot },
            BinaryOp::Wedge => quote! { crate::Wedge },
        }
    }

    pub fn trait_fn(&self) -> TokenStream {
        match self {
            BinaryOp::Mul => quote! { mul },
            BinaryOp::Dot => quote! { dot },
            BinaryOp::Wedge => quote! { wedge },
        }
    }
}

impl ImplBinaryOp {
    fn output(&self) -> Type {
        let iter = self
            .lhs
            .blades()
            .flat_map(|lhs_b| {
                self.rhs
                    .blades()
                    .map(move |rhs_b| self.op.call(lhs_b, rhs_b))
            })
            .filter_map(|product| product.blade());
        Type::from_iter(iter, self.lhs.algebra())
    }

    fn expr(&self) -> TokenStream {
        let output = self.output();

        let is_scalar = match output {
            Type::Grade(grade) => grade.is_scalar(),
            _ => false,
        };

        let (ty, blades) = &match self.output() {
            Type::Zero(_) => return quote! { crate::Zero },
            Type::Grade(grade) => {
                let ty = grade.type_ident();
                let blades = grade.blades().collect::<Vec<_>>();
                (ty, blades)
            }
            Type::SubAlgebra(sub) => {
                let ty = sub.type_ident();
                let blades = sub.blades().collect::<Vec<_>>();
                (ty, blades)
            }
        };

        let fields = blades.iter().map(|&blade| {
            let products = self
                .lhs
                .blades()
                .flat_map(|lhs| {
                    self.rhs.blades().map(move |rhs| {
                        let product = self.op.call(lhs, rhs);
                        (lhs, rhs, product)
                    })
                })
                .filter_map(|(lhs, rhs, product)| match product {
                    Product::Pos(b) | Product::Neg(b) if blade == b => {
                        let lhs = lhs.field();
                        let rhs = rhs.field();
                        let expr = quote! { self.#lhs * rhs.#rhs };

                        if product.is_neg() {
                            Some(quote! { -(#expr) })
                        } else {
                            Some(expr)
                        }
                    }
                    _ => None,
                })
                .collect::<syn::punctuated::Punctuated<_, syn::token::Add>>();

            let expr = if products.is_empty() {
                quote! { 0. }
            } else {
                quote! { #products }
            };

            if is_scalar {
                expr
            } else {
                let f = blade.field();
                quote! { #f: #expr, }
            }
        });

        if is_scalar {
            quote! {
                #( #fields )*
            }
        } else {
            quote! {
                #ty {
                    #( #fields )*
                }
            }
        }
    }
}

impl ToTokens for ImplBinaryOp {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let ty = self.lhs.type_ident();
        let rhs_ty = self.rhs.type_ident();
        let trait_ty = self.op.trait_ty();
        let trait_fn = self.op.trait_fn();

        let output = self.output();

        let output_ty = output.type_ident();
        let expr = self.expr();
        let rhs_ident = match output {
            Type::Zero(_) => quote! { _ },
            _ => quote! { rhs },
        };

        let op_impl = quote! {
            impl const #trait_ty<#rhs_ty> for #ty {
                type Output = #output_ty;
                fn #trait_fn(self, #rhs_ident: #rhs_ty) -> Self::Output {
                    #expr
                }
            }
        };

        tokens.extend(op_impl)
    }
}

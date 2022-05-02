use super::types::*;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use syn::punctuated::Punctuated;
use syn::token::Add;

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

        let unary_ops = UnaryOp::iter().map(|op| ImplUnaryOp { ty: *self, op });

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

            #(#unary_ops)*
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

impl Type {
    fn is_scalar(&self) -> bool {
        match self {
            Self::Grade(grade) => grade.is_scalar(),
            _ => false,
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

        let (ty, blades) = &match output {
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
                .collect::<Punctuated<_, Add>>();

            let expr = if products.is_empty() {
                quote! { 0. }
            } else {
                quote! { #products }
            };

            if output.is_scalar() {
                expr
            } else {
                let f = blade.field();
                quote! { #f: #expr, }
            }
        });

        if output.is_scalar() {
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

struct ImplUnaryOp {
    pub ty: Type,
    pub op: UnaryOp,
}

impl ToTokens for ImplUnaryOp {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let ty = self.ty.type_ident();
        let output_blades = self.ty.blades().filter_map(|b| self.op.call(b).blade());
        let output = Type::from_iter(output_blades, self.ty.algebra());
        let output_ty = output.type_ident();

        let op_trait = self.op.trait_ty();
        let op_fn = self.op.trait_fn();

        let fields = output.blades().map(|blade| {
            let sum = self
                .ty
                .blades()
                .filter_map(|b| {
                    let f = b.field();
                    let product = self.op.call(b);
                    let product_blade = product.blade()?;

                    if product_blade == blade {
                        let expr = if product.is_neg() {
                            quote! { -self.#f }
                        } else {
                            quote! { self.#f }
                        };

                        Some(expr)
                    } else {
                        None
                    }
                })
                .collect::<Punctuated<_, Add>>();

            let expr = if sum.is_empty() {
                quote! { 0. }
            } else {
                quote! { #sum }
            };

            if output.is_scalar() {
                quote! { #expr }
            } else {
                let f = blade.field();
                quote! { #f: #expr }
            }
        });

        let expr = if self.ty.is_scalar() {
            quote! { #(#fields)* }
        } else {
            quote! { #ty { #(#fields,)* } }
        };

        let t = quote! {
            impl #op_trait for #ty {
                type Output = #output_ty;
                fn #op_fn(self) -> Self::Output {
                    #expr
                }
            }
        };

        tokens.extend(t);
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum UnaryOp {
    Neg,
    Reverse,
    Antireverse,
}

impl UnaryOp {
    fn iter() -> impl Iterator<Item = Self> {
        [Self::Neg, Self::Reverse, Self::Antireverse].into_iter()
    }

    fn trait_ty(&self) -> TokenStream {
        match self {
            Self::Neg => quote! { std::ops::Neg },
            Self::Reverse => quote! { crate::Reverse },
            Self::Antireverse => quote! { crate::Antireverse },
        }
    }

    fn trait_fn(&self) -> TokenStream {
        match self {
            Self::Neg => quote! { neg },
            Self::Reverse => quote! { rev },
            Self::Antireverse => quote! { antirev },
        }
    }

    fn call(&self, blade: Blade) -> Product {
        match self {
            Self::Neg => Product::Neg(blade),
            Self::Reverse => {
                let grade = blade.grade().0;
                if (grade / 2) % 2 == 0 {
                    Product::Pos(blade)
                } else {
                    Product::Neg(blade)
                }
            }
            Self::Antireverse => {
                let antiscalar = blade.1.pseudoscalar();
                let set = antiscalar.0 .0 ^ blade.0 .0;
                let anti = blade.1.blade(set);
                match Self::Reverse.call(anti) {
                    Product::Zero => Product::Zero,
                    Product::Pos(_) => Product::Pos(blade),
                    Product::Neg(_) => Product::Neg(blade),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse() {
        let op = UnaryOp::Reverse;
        let alg = Algebra::new(4, 0, 0);
        let scalar = Blade(BladeSet(0), alg);
        let vector = Blade(BladeSet(0b_1), alg);
        let bivector = Blade(BladeSet(0b_11), alg);
        let trivector = Blade(BladeSet(0b_111), alg);
        let quadvector = Blade(BladeSet(0b_1111), alg);

        assert!(matches!(op.call(scalar), Product::Pos(_)));
        assert!(matches!(op.call(vector), Product::Pos(_)));
        assert!(matches!(op.call(bivector), Product::Neg(_)));
        assert!(matches!(op.call(trivector), Product::Neg(_)));
        assert!(matches!(op.call(quadvector), Product::Pos(_)));
    }

    #[test]
    fn antireverse() {
        let op = UnaryOp::Antireverse;
        let alg = Algebra::new(4, 0, 0);
        let scalar = Blade(BladeSet(0), alg);
        let vector = Blade(BladeSet(0b_1), alg);
        let bivector = Blade(BladeSet(0b_11), alg);
        let trivector = Blade(BladeSet(0b_111), alg);
        let quadvector = Blade(BladeSet(0b_1111), alg);

        assert!(matches!(op.call(scalar), Product::Pos(_)));
        assert!(matches!(op.call(vector), Product::Neg(_)));
        assert!(matches!(op.call(bivector), Product::Neg(_)));
        assert!(matches!(op.call(trivector), Product::Pos(_)));
        assert!(matches!(op.call(quadvector), Product::Pos(_)));
    }
}

use super::types::*;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};

impl Algebra {
    pub fn define(&self) -> TokenStream {
        let traits = traits();
        let zero = zero();

        let blades = self
            .blades()
            .filter(|b| !b.0.is_empty())
            .map(|b| b.define());

        let grades = self.grades_without_scalar().map(|g| g.define());

        let subalgebra = self.subalgebras().map(|s| s.define());

        quote! {
            #traits

            #zero

            #( #blades )*

            #( #grades )*

            #( #subalgebra )*
        }
    }
}

fn zero() -> TokenStream {
    quote! {
        // TODO check that type paths are correct according to best practices
        pub use crate::Zero;
    }
}

fn traits() -> TokenStream {
    quote! {
        // TODO specifty paths explicitly in procedural code
        pub use crate::{Dot, Wedge, Commutator};
    }
}

impl Blade {
    pub fn define(&self) -> proc_macro2::TokenStream {
        let ty = self.type_ident();

        let mul_blades = self.1.blades().filter(|b| b.0 .0 != 0).map(|rhs| {
            let rhs_ty = rhs.type_ident();
            let product = *self * rhs;

            let output_ty = match product.blade() {
                Some(blade) => {
                    let ty = blade.type_ident();
                    quote! { #ty }
                }
                None => quote! { crate::Zero },
            };

            let expr = match product {
                Product::Zero => quote! { crate::Zero },
                Product::Pos(b) => {
                    if b == self.1.scalar() {
                        quote! { self.0 * rhs.0 }
                    } else {
                        quote! { #output_ty(self.0 * rhs.0) }
                    }
                }
                Product::Neg(b) => {
                    if b == self.1.scalar() {
                        quote! { -(self.0 * rhs.0) }
                    } else {
                        quote! { #output_ty(-(self.0 * rhs.0)) }
                    }
                }
            };

            let rhs_ident = match product {
                Product::Zero => quote! { _ },
                _ => quote! { rhs },
            };

            quote! {
                impl const std::ops::Mul<#rhs_ty> for #ty {
                    type Output = #output_ty;
                    fn mul(self, #rhs_ident: #rhs_ty) -> Self::Output {
                        #expr
                    }
                }
            }
        });

        quote! {
            #[derive(Default, Copy, Clone, PartialEq, PartialOrd)]
            pub struct #ty(f64);

            impl const From<crate::Zero> for #ty {
                fn from(_: crate::Zero) -> #ty {
                    #ty(0.)
                }
            }

            impl std::fmt::Debug for #ty {
                fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(f, "{:?}", self.0)
                }
            }

            impl const std::ops::Neg for #ty {
                type Output = #ty;
                fn neg(self) -> #ty {
                    Self(-self.0)
                }
            }

            impl const std::ops::Mul<f64> for #ty {
                type Output = #ty;
                fn mul(self, rhs: f64) -> #ty {
                    #ty(self.0 * rhs)
                }
            }

            impl const std::ops::Mul<#ty> for f64 {
                type Output = #ty;
                fn mul(self, rhs: #ty) -> #ty {
                    #ty(self * rhs.0)
                }
            }

            impl const std::ops::Mul<crate::Zero> for #ty {
                type Output = crate::Zero;
                fn mul(self, rhs: crate::Zero) -> crate::Zero {
                    rhs
                }
            }

            impl const std::ops::Div<f64> for #ty {
                type Output = #ty;
                fn div(self, rhs: f64) -> #ty {
                    #ty(self.0 / rhs)
                }
            }

            #(#mul_blades)*

            impl const std::ops::Add<crate::Zero> for #ty {
                type Output = #ty;
                fn add(self, _rhs: crate::Zero) -> #ty {
                    self
                }
            }

            impl const std::ops::Sub<crate::Zero> for #ty {
                type Output = #ty;
                fn sub(self, _rhs: crate::Zero) -> #ty {
                    self
                }
            }

            impl const std::ops::Add for #ty {
                type Output = #ty;
                fn add(self, rhs: #ty) -> #ty {
                    #ty(self.0 + rhs.0)
                }
            }

            impl const std::ops::Sub for #ty {
                type Output = #ty;
                fn sub(self, rhs: #ty) -> #ty {
                    #ty(self.0 - rhs.0)
                }
            }
        }
    }

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

    pub fn type_ident(&self) -> Ident {
        if self.0.is_empty() {
            Ident::new("f64", Span::mixed_site())
        } else {
            let mut output = "E".to_string();

            for basis in self.1.bases() {
                if self.0.contains(basis.0) {
                    output.push_str(&basis.0.to_string());
                }
            }

            Ident::new(&output, Span::mixed_site())
        }
    }
}

impl Grade {
    pub fn define(&self) -> TokenStream {
        let ty = self.type_ident();

        let struct_fields = self.blades().map(|b| {
            let f = b.field();
            let ty = b.type_ident();
            quote! { pub #f: #ty, }
        });

        let new_fn_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: f64 }
        });

        let new_fn_struct_fields = self.blades().map(|b| {
            let f = b.field();
            let ty = b.type_ident();
            quote! { #f: #ty(#f), }
        });

        let zero_fields = self.blades().map(|b| {
            let f = b.field();
            let ty = b.type_ident();
            quote! { #f: #ty::from(crate::Zero), }
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

        let alg = self.algebra();
        let type_ops = alg.types().flat_map(|t| {
            BinaryOp::iter().map(move |op| {
                ImplementBinaryOp {
                    lhs: self.as_type(),
                    rhs: t,
                    op,
                }
                .implement()
            })
        });

        quote! {
            #[derive(Debug, Default, Copy, Clone, PartialEq)]
            pub struct #ty {
                #(#struct_fields)*
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

            impl const From<crate::Zero> for #ty {
                fn from(_: crate::Zero) -> #ty {
                    Self {
                        #(#zero_fields)*
                    }
                }
            }

            impl const std::ops::Neg for #ty {
                type Output = #ty;
                fn neg(self) -> Self {
                    Self {
                        #(#neg_fields)*
                    }
                }
            }

            impl const std::ops::Add for #ty {
                type Output = Self;
                fn add(self, rhs: Self) -> Self {
                    Self {
                        #(#add_self_fields)*
                    }
                }
            }

            impl const std::ops::Add<crate::Zero> for #ty {
                type Output = Self;
                fn add(self, _rhs: crate::Zero) -> Self {
                    self
                }
            }

            impl const std::ops::Sub for #ty {
                type Output = Self;
                fn sub(self, rhs: Self) -> Self {
                    Self {
                        #(#sub_self_fields)*
                    }
                }
            }

            impl const std::ops::Sub<crate::Zero> for #ty {
                type Output = Self;
                fn sub(self, _rhs: crate::Zero) -> Self {
                    self
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

            impl const std::ops::Div<f64> for #ty {
                type Output = Self;
                fn div(self, rhs: f64) -> Self {
                    Self {
                        #(#div_f64_fields)*
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

            #(#type_ops)*
        }
    }

    pub fn type_ident(&self) -> Ident {
        let str = match self.0 {
            0 => "f64",
            1 => "Vector",
            2 => "Bivector",
            3 => "Trivector",
            4 => "Quadvector",
            5 => "Pentavector",
            6 => "Hexavector",
            7 => "Heptavector",
            8 => "Octovector",
            9 => "Nonavector",
            _ => unimplemented!("not implemented for grade: {}", self.0),
        };
        Ident::new(str, Span::mixed_site())
    }
}

impl SubAlgebra {
    pub fn define(&self) -> TokenStream {
        let ty = match self {
            SubAlgebra::Even(_) => quote! { Even },
            SubAlgebra::Odd(_) => quote! { Odd },
        };

        let blades = self.blades().map(|b| {
            let f = b.field();
            let ty = b.type_ident();
            quote! { pub #f: #ty, }
        });

        let new_fn_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: f64 }
        });

        let new_fn_struct_fields = self.blades().map(|b| {
            let f = b.field();
            let ty = b.type_ident();
            if b == self.algebra().scalar() {
                quote! { #f, }
            } else {
                quote! { #f: #ty(#f), }
            }
        });

        let zero_fields = self.blades().map(|b| {
            let f = b.field();
            let ty = b.type_ident();
            quote! { #f: #ty::from(crate::Zero), }
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

        let type_ops = self.algebra().types().into_iter().flat_map(|rhs| {
            BinaryOp::iter().map(move |op| {
                ImplementBinaryOp {
                    lhs: self.as_type(),
                    rhs,
                    op,
                }
                .implement()
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

            impl const From<crate::Zero> for #ty {
                fn from(_: crate::Zero) -> #ty {
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

            impl const std::ops::Add<crate::Zero> for #ty {
                type Output = #ty;
                fn add(self, _rhs: crate::Zero) -> Self::Output {
                    self
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

            impl const std::ops::Sub<crate::Zero> for #ty {
                type Output = #ty;
                fn sub(self, _rhs: crate::Zero) -> Self::Output {
                    self
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
}

pub struct ImplementBinaryOp {
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

impl ImplementBinaryOp {
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
        let (ty, blades) = match self.output() {
            Type::Zero(_) => return quote! { crate::Zero },
            Type::Grade(grade) => {
                let blades = grade.blades().collect::<Vec<_>>();
                if grade.0 == 0 {
                    (None, blades)
                } else {
                    let ty = grade.type_ident();
                    (Some(ty), blades)
                }
            }
            Type::SubAlgebra(sub) => {
                let ty = sub.type_ident();
                let blades = sub.blades().collect::<Vec<_>>();
                (Some(ty), blades)
            }
        };

        let ty_some = ty.is_some();
        let fields = blades.into_iter().map(|b| {
            let f = b.field();

            let products = self
                .lhs
                .blades()
                .flat_map(|lhs_b| {
                    self.rhs.blades().map(move |rhs_b| {
                        let product = self.op.call(lhs_b, rhs_b);
                        (lhs_b, rhs_b, product)
                    })
                })
                .filter(|(_, _, product)| {
                    if let Some(blade) = product.blade() {
                        blade == b
                    } else {
                        false
                    }
                })
                .map(|(lhs_b, rhs_b, _)| {
                    let lhs_f = lhs_b.field();
                    let rhs_f = rhs_b.field();
                    quote! { self.#lhs_f * rhs.#rhs_f }
                })
                .collect::<syn::punctuated::Punctuated<_, syn::token::Add>>();

            if ty_some {
                if products.is_empty() {
                    if ty_some {
                        quote! { #f: crate::Zero.into(), }
                    } else {
                        quote! { crate::Zero.into(), }
                    }
                } else {
                    if ty_some {
                        quote! { #f: #products, }
                    } else {
                        quote! { #products }
                    }
                }
            } else {
                quote! { #products }
            }
        });

        if let Some(ty) = ty {
            quote! {
                #ty {
                    #( #fields )*
                }
            }
        } else {
            quote! {
                #( #fields )*
            }
        }
    }

    pub fn implement(&self) -> TokenStream {
        let ty = self.lhs.type_ident();
        let rhs_ty = self.rhs.type_ident();
        let trait_ty = self.op.trait_ty();
        let trait_fn = self.op.trait_fn();
        let output_ty = self.output().type_ident();
        let expr = self.expr();
        let rhs_ident = match self.output() {
            Type::Zero(_) => quote! { _ },
            _ => quote! { rhs },
        };
        quote! {
            impl const #trait_ty<#rhs_ty> for #ty {
                type Output = #output_ty;
                fn #trait_fn(self, #rhs_ident: #rhs_ty) -> Self::Output {
                    #expr
                }
            }
        }
    }
}

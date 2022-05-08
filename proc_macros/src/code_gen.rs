use super::types::*;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use syn::punctuated::Punctuated;
use syn::token::Add;

impl Algebra {
    pub fn define(&self) -> TokenStream {
        let traits = if self.is_homogenous() {
            quote! {
                pub use crate::{
                    Dot,
                    Antidot,
                    Wedge,
                    Antiwedge,
                    Commutator,
                    Geometric,
                    Antigeometric,
                    LeftComplement,
                    RightComplement,
                    Reverse,
                    Antireverse,
                    Bulk,
                    Weight,
                    IsIdeal
                };
            }
        } else {
            quote! {
                pub use crate::{
                    Dot,
                    Wedge,
                    Commutator,
                    Geometric,
                    LeftComplement,
                    RightComplement,
                    Reverse
                };
            }
        };
        let types = self.types().map(|ty| ty.define());

        quote! {
            #traits
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
        let str = if self.is_pseudoscalar() {
            "Pseudoscalar"
        } else {
            match self.0 {
                0 => "f64",
                1 => "Vector",
                2 => "Bivector",
                3 => "Trivector",
                4 => "Quadvector",
                5 => "Pentavector",
                6 => "Hexavector",
                _ => unimplemented!("not implemented for grade: {}", self.0),
            }
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
            Type::Zero(_) => Zero::ty(),
            Type::Grade(grade) => grade.type_ident().to_token_stream(),
            Type::SubAlgebra(sub) => sub.type_ident().to_token_stream(),
        }
    }

    pub fn define(&self) -> TokenStream {
        let ty = self.type_ident();

        let definition = if self.is_local() {
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

            let default_fields = self.blades().map(|b| {
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

            let div_f64 = impl_div_f64(*self, &ty);

            quote! {
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct #ty {
                    #( #blades )*
                }

                impl #ty {
                    #[inline]
                    pub const fn new(
                        #(#new_fn_fields,)*
                    ) -> Self {
                        Self {
                            #( #new_fn_struct_fields )*
                        }
                    }
                }

                impl const Default for #ty {
                    #[inline]
                    fn default() -> Self {
                        Self {
                            #( #default_fields )*
                        }
                    }
                }

                impl const std::ops::Add for #ty {
                    type Output = #ty;
                    #[inline]
                    fn add(self, rhs: Self) -> Self {
                        Self {
                            #( #add_self_fields )*
                        }
                    }
                }

                impl const std::ops::Sub for #ty {
                    type Output = #ty;
                    #[inline]
                    fn sub(self, rhs: Self) -> Self {
                        Self {
                            #( #sub_self_fields )*
                        }
                    }
                }

                #div_f64
            }
        } else {
            quote! {}
        };

        let algebra = self.algebra();
        let product_ops = algebra.types().flat_map(|rhs| {
            ProductOps::iter().map(move |op| ImplProductOp {
                lhs: *self,
                rhs,
                op,
            })
        });

        let unary_ops = UnaryOp::iter().map(|op| ImplUnaryOp { ty: *self, op });

        quote! {
            #definition

            #( #product_ops )*

            #( #unary_ops )*
        }
    }
}

fn impl_div_f64(lhs: Type, ty: &TokenStream) -> TokenStream {
    let fields = lhs.blades().map(|b| {
        let f = b.field();
        let lhs = access_value(lhs, b, quote!(self));
        quote! { #f: #lhs / rhs }
    });

    let expr = construct_output(lhs, fields);

    quote! {
        impl const std::ops::Div<f64> for #ty {
            type Output = Self;
            #[inline]
            fn div(self, rhs: f64) -> Self {
                #expr
            }
        }
    }
}

fn access_value(parent: Type, blade: Blade, ident: TokenStream) -> TokenStream {
    if parent.is_scalar() {
        ident
    } else {
        let field = blade.field();
        quote! {
            #ident.#field
        }
    }
}

fn construct_output<F: Iterator<Item = TokenStream>>(output: Type, fields: F) -> TokenStream {
    if output.is_zero() {
        output.type_ident()
    } else if output.is_scalar() {
        quote! { #( #fields )* }
    } else {
        let ty = output.type_ident();
        quote! {
            #ty {
                #( #fields, )*
            }
        }
    }
}

fn assign_value(output: Type, blade: Blade, sum: Punctuated<TokenStream, Add>) -> TokenStream {
    let expr = if sum.is_empty() {
        quote! { 0. }
    } else {
        sum.to_token_stream()
    };
    if output.is_scalar() {
        expr
    } else {
        let field = blade.field();
        quote! { #field: #expr }
    }
}

pub struct ImplProductOp {
    pub lhs: Type,
    pub rhs: Type,
    pub op: ProductOps,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ProductOps {
    Mul,
    Geometric,
    Dot,
    Wedge,
}

impl ProductOps {
    pub fn iter() -> impl Iterator<Item = Self> + 'static {
        [Self::Mul, Self::Geometric, Self::Dot, Self::Wedge].into_iter()
    }

    pub fn call(&self, lhs: Blade, rhs: Blade) -> Product {
        match self {
            ProductOps::Mul => lhs * rhs,
            ProductOps::Geometric => lhs * rhs,
            ProductOps::Dot => lhs.dot(rhs),
            ProductOps::Wedge => lhs.wedge(rhs),
        }
    }

    pub fn trait_ty(&self) -> TokenStream {
        match self {
            ProductOps::Mul => quote! { std::ops::Mul },
            ProductOps::Geometric => quote! { crate::Geometric },
            ProductOps::Dot => quote! { crate::Dot },
            ProductOps::Wedge => quote! { crate::Wedge },
        }
    }

    pub fn trait_fn(&self) -> TokenStream {
        match self {
            ProductOps::Mul => quote! { mul },
            ProductOps::Geometric => quote! { geo },
            ProductOps::Dot => quote! { dot },
            ProductOps::Wedge => quote! { wedge },
        }
    }
}

impl ImplProductOp {
    fn output(&self) -> Type {
        Type::from_iter(
            self.lhs.blades().flat_map(|lhs| {
                self.rhs
                    .blades()
                    .filter_map(move |rhs| self.op.call(lhs, rhs).blade())
            }),
            self.lhs.algebra(),
        )
    }

    fn expr(&self, output: Type) -> TokenStream {
        let fields = output.blades().map(|blade| {
            let sum = self
                .lhs
                .blades()
                .flat_map(|lhs| {
                    self.rhs.blades().map(move |rhs| {
                        let product = self.op.call(lhs, rhs);
                        (lhs, rhs, product)
                    })
                })
                .filter_map(|(lhs, rhs, product)| {
                    if product.blade() == Some(blade) {
                        let lhs = access_value(self.lhs, lhs, quote!(self));
                        let rhs = access_value(self.rhs, rhs, quote!(rhs));
                        let expr = quote! { #lhs * #rhs };

                        if product.is_neg() {
                            Some(quote! { -(#expr) })
                        } else {
                            Some(expr)
                        }
                    } else {
                        None
                    }
                })
                .collect::<Punctuated<_, Add>>();

            assign_value(output, blade, sum)
        });

        construct_output(output, fields)
    }
}

impl ToTokens for ImplProductOp {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        // product traits for f64 and f64 are implemented in the clifford crate
        if !self.lhs.is_local() && !self.rhs.is_local() {
            return;
        }

        let ty = self.lhs.type_ident();
        let rhs_ty = self.rhs.type_ident();
        let trait_ty = self.op.trait_ty();
        let trait_fn = self.op.trait_fn();

        let output = self.output();

        let output_ty = output.type_ident();
        let expr = self.expr(output);
        let rhs_ident = if output.is_zero() {
            quote! { _ }
        } else {
            quote! { rhs }
        };

        let op_impl = quote! {
            impl const #trait_ty<#rhs_ty> for #ty {
                type Output = #output_ty;
                #[inline]
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
        if !self.ty.is_local() {
            return;
        }

        if matches!(self.op, UnaryOp::Bulk | UnaryOp::Weight) && !self.ty.algebra().is_homogenous()
        {
            return;
        }

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
                    let product = self.op.call(b);
                    if product.blade() == Some(blade) {
                        let value = access_value(self.ty, b, quote!(self));
                        let expr = if product.is_neg() {
                            quote! { -#value }
                        } else {
                            quote! { #value }
                        };
                        Some(expr)
                    } else {
                        None
                    }
                })
                .collect::<Punctuated<_, Add>>();

            assign_value(output, blade, sum)
        });

        let expr = construct_output(output, fields);

        let t = quote! {
            impl #op_trait for #ty {
                type Output = #output_ty;
                #[inline]
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
    LeftComplement,
    RightComplement,
    Bulk,
    Weight,
}

impl UnaryOp {
    fn iter() -> impl Iterator<Item = Self> {
        [
            Self::Neg,
            Self::Reverse,
            Self::LeftComplement,
            Self::RightComplement,
            Self::Bulk,
            Self::Weight,
        ]
        .into_iter()
    }

    fn trait_ty(&self) -> TokenStream {
        match self {
            Self::Neg => quote! { std::ops::Neg },
            Self::Reverse => quote! { crate::Reverse },
            Self::LeftComplement => quote! { crate::LeftComplement },
            Self::RightComplement => quote! { crate::RightComplement },
            Self::Bulk => quote! { crate::Bulk },
            Self::Weight => quote! { crate::Weight },
        }
    }

    fn trait_fn(&self) -> TokenStream {
        match self {
            Self::Neg => quote! { neg },
            Self::Reverse => quote! { rev },
            Self::LeftComplement => quote! { left_comp },
            Self::RightComplement => quote! { right_comp },
            Self::Bulk => quote! { bulk },
            Self::Weight => quote! { weight },
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
            Self::LeftComplement => {
                let antiscalar = blade.1.pseudoscalar();
                let set = antiscalar.0 .0 ^ blade.0 .0;
                let complement = blade.1.blade(set);
                (complement * blade).with_blade(complement)
            }
            Self::RightComplement => {
                let antiscalar = blade.1.pseudoscalar();
                let set = antiscalar.0 .0 ^ blade.0 .0;
                let complement = blade.1.blade(set);
                (blade * complement).with_blade(complement)
            }
            Self::Bulk => {
                let null_blade = blade.1.null_basis().unwrap();
                match blade.wedge(null_blade) {
                    Product::Zero => Product::Zero,
                    _ => Product::Pos(blade),
                }
            }
            Self::Weight => {
                let null_blade = blade.1.null_basis().unwrap();
                match blade.wedge(null_blade) {
                    Product::Zero => Product::Pos(blade),
                    _ => Product::Zero,
                }
            }
        }
    }
}

struct SandwichProduct {
    lhs: Type,
    rhs: Type,
}

impl ToTokens for SandwichProduct {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let lhs_ty = self.lhs.type_ident();
        let rhs_ty = self.rhs.type_ident();

        let _intermediate = ImplProductOp {
            lhs: self.lhs,
            rhs: self.rhs,
            op: ProductOps::Geometric,
        }
        .output();

        let t = quote! {
            impl crate::Sandwich<#rhs_ty> for #lhs_ty {
                type Output = #rhs_ty;
                fn sandwich(self, rhs: #rhs_ty) -> Self::Output {
                    let int = self.geo(rhs);
                    todo!()
                }
            }
        };

        tokens.extend(t);
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
    fn left_comp() {
        let left_comp = |blade: Blade| UnaryOp::LeftComplement.call(blade);
        let pga = Algebra::new(3, 0, 1);
        let s = pga.blade(0);
        let e1 = pga.blade(0b0001);
        let e12 = pga.blade(0b0011);
        let e123 = pga.blade(0b0111);
        let e1234 = pga.blade(0b1111);
        let e234 = pga.blade(0b1110);
        let e34 = pga.blade(0b1100);
        let e4 = pga.blade(0b1000);

        assert_eq!(Product::Pos(e1234), left_comp(s));
        assert_eq!(Product::Neg(e234), left_comp(e1));
        assert_eq!(Product::Pos(e34), left_comp(e12));
        assert_eq!(Product::Neg(e4), left_comp(e123));
        assert_eq!(Product::Pos(s), left_comp(e1234));
    }

    #[test]
    fn right_comp() {
        let right_comp = |blade: Blade| UnaryOp::RightComplement.call(blade);
        let pga = Algebra::new(3, 0, 1);
        let s = pga.blade(0);
        let e1 = pga.blade(0b0001);
        let e12 = pga.blade(0b0011);
        let e123 = pga.blade(0b0111);
        let e1234 = pga.blade(0b1111);
        let e234 = pga.blade(0b1110);
        let e34 = pga.blade(0b1100);
        let e4 = pga.blade(0b1000);

        assert_eq!(Product::Pos(e1234), right_comp(s));
        assert_eq!(Product::Pos(e234), right_comp(e1));
        assert_eq!(Product::Pos(e34), right_comp(e12));
        assert_eq!(Product::Pos(e4), right_comp(e123));
        assert_eq!(Product::Pos(s), right_comp(e1234));
    }
}

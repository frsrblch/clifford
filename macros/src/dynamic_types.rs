use crate::algebra::{Complement, InverseOps, NormOps, ProductOp, SumOp, Type};
use crate::code_gen::{GradeProduct, Reverse, Sqrt};
use crate::Algebra;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{parse_quote, ItemEnum, ItemImpl};

impl Algebra {
    pub fn dynamic_types(self) -> TokenStream {
        let define_enum = self.define_enum();
        let binary_op = self.define_binary_op();
        let unary_op = self.define_unary_op();
        let product_ops = self.define_product_ops();
        let grade_products = self.impl_grade_products();
        let sum_ops = self.define_sum_ops();
        let norm_ops = self.define_norm_ops();
        let inv_ops = self.define_inv_ops();
        let sqrt_ops = self.define_sqrt();
        let dual_ops = self.define_duals();
        let rev_ops = self.define_rev();
        let neg_ops = self.define_neg();
        quote! {
            #define_enum
            #binary_op
            #(#product_ops)*
            #(#grade_products)*
            #(#sum_ops)*
            #(#norm_ops)*
            #(#inv_ops)*
            #sqrt_ops
            #(#dual_ops)*
            #rev_ops
            #neg_ops
            #unary_op
        }
    }

    fn define_enum(self) -> ItemEnum {
        let variants = self.types().map(|ty| {
            quote! {
                #ty(#ty<T>),
            }
        });
        parse_quote! {
            #[derive(Debug, Copy, Clone, Eq, PartialEq)]
            pub enum Value<T> {
                #(#variants)*
            }
        }
    }

    fn define_binary_op(self) -> TokenStream {
        let len = proc_macro2::Literal::usize_unsuffixed(8 + self.grades().count());
        let grade_values = self.grades();
        let grade_products = self.grades();
        let call_variants = BinaryOp::iter(self).map(|op| match op {
            BinaryOp::Add => quote! { BinaryOp::Add => std::ops::Add::add(lhs, rhs), },
            BinaryOp::Sub => quote! { BinaryOp::Sub => std::ops::Sub::sub(lhs, rhs), },
            BinaryOp::Geo => quote! { BinaryOp::Geo => Geo::geo(lhs, rhs), },
            BinaryOp::Dot => quote! { BinaryOp::Dot => Dot::dot(lhs, rhs), },
            BinaryOp::Wedge => quote! { BinaryOp::Wedge => Wedge::wedge(lhs, rhs), },
            BinaryOp::Antigeo => quote! { BinaryOp::Antigeo => Antigeo::antigeo(lhs, rhs), },
            BinaryOp::Antidot => quote! { BinaryOp::Antidot => Antidot::antidot(lhs, rhs), },
            BinaryOp::Antiwedge => {
                quote! { BinaryOp::Antiwedge => Antiwedge::antiwedge(lhs, rhs), }
            }
            BinaryOp::Grade(g) => {
                let grade = Type::Grade(g);
                quote! {
                    BinaryOp::#grade => #grade::product(lhs, rhs),
                }
            }
        });
        quote! {
            #[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
            enum BinaryOp {
                Add,
                Sub,
                Geo,
                Dot,
                Wedge,
                Antigeo,
                Antidot,
                Antiwedge,
                #(#grade_products,)*
            }

            impl BinaryOp {
                pub fn values() -> [Self; #len] {
                    use BinaryOp::*;
                    [
                        Add, Sub, Geo, Dot, Wedge, Antigeo, Antidot, Antiwedge,
                        #(#grade_values,)*
                    ]
                }

                pub fn len() -> usize {
                    Self::values().len()
                }

                pub fn call<T>(self, lhs: Value<T>, rhs: Value<T>) -> Option<Value<T>>
                where
                    T: std::ops::Mul<Output = T>
                        + std::ops::Add<Output = T>
                        + std::ops::Sub<Output = T>
                        + std::ops::Neg<Output = T>
                        + num_traits::Zero
                        + Copy,
                {
                    match self {
                        #(#call_variants)*
                    }
                }
            }
        }
    }

    fn impl_grade_products(self) -> impl Iterator<Item = ItemImpl> {
        let trait_ty = GradeProduct::trait_ty();
        let trait_fn = GradeProduct::trait_fn();
        self.grades().map(move |grade| {
            let match_variants = self.type_tuples().filter_map(|(lhs, rhs)| {
                if GradeProduct::contains(lhs, rhs, grade, self) {
                    Some(quote! {
                         (Value::#lhs(lhs), Value::#rhs(rhs)) => Some(Value::#grade(#grade::<T>::#trait_fn(lhs, rhs))),
                    })
                } else {
                    None
                }
            });
            parse_quote! {
                impl<T> #trait_ty<Value<T>, Value<T>> for #grade<T>
                where
                    T: std::ops::Mul<Output = T>
                        + std::ops::Add<Output = T>
                        + std::ops::Sub<Output = T>
                        + std::ops::Neg<Output = T>
                        + num_traits::Zero
                        + Copy,
                {
                    type Output = Option<Value<T>>;
                    #[inline]
                    fn #trait_fn(lhs: Value<T>, rhs: Value<T>) -> Self::Output {
                        match (lhs, rhs) {
                            #(#match_variants)*
                            _ => None,
                        }
                    }
                }
            }
        })
    }

    fn define_unary_op(self) -> TokenStream {
        let dual_variants = Complement::iter(self).map(|op| op.trait_ty());

        let call_variants = UnaryOp::iter(self).map(|op| match op {
            UnaryOp::Norm2 => quote! { UnaryOp::Norm2 => value.norm2() },
            UnaryOp::Norm => quote! { UnaryOp::Norm => value.norm() },
            UnaryOp::Unit => quote! { UnaryOp::Unit => value.unit() },
            UnaryOp::Inv => quote! { UnaryOp::Inv => value.inv() },
            UnaryOp::Sqrt => quote! { UnaryOp::Sqrt => num_sqrt::Sqrt::sqrt(value) },
            UnaryOp::Dual => quote! { UnaryOp::Dual => Some(value.dual()) },
            UnaryOp::LeftComp => quote! { UnaryOp::LeftComp => Some(value.left_comp()) },
            UnaryOp::RightComp => quote! { UnaryOp::RightComp => Some(value.right_comp()) },
            UnaryOp::Rev => quote! { UnaryOp::Rev => Some(value.rev()) },
            UnaryOp::Neg => quote! { UnaryOp::Neg => Some(-value) },
        });

        let fn_values = if self.symmetrical_complements() {
            quote! {
                pub fn values() -> [Self; 8] {
                    use UnaryOp::*;
                    [Norm2, Norm, Unit, Inv, Sqrt, Dual, Rev, Neg]
                }
            }
        } else {
            quote! {
                pub fn values() -> [Self; 9] {
                    use UnaryOp::*;
                    [Norm2, Norm, Unit, Inv, Sqrt, LeftComp, RightComp, Rev, Neg]
                }
            }
        };

        quote! {
            #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
            pub enum UnaryOp {
                Norm2,
                Norm,
                Unit,
                Inv,
                Sqrt,
                #(#dual_variants,)*
                Rev,
                Neg,
            }

            impl UnaryOp {
                #fn_values

                pub fn len() -> usize {
                    Self::values().len()
                }

                pub fn call<T>(self, value: Value<T>) -> Option<Value<T>>
                where
                    T: std::ops::Mul<Output = T>
                        + std::ops::Div<Output = T>
                        + std::ops::Add<Output = T>
                        + std::ops::Sub<Output = T>
                        + std::ops::Neg<Output = T>
                        + PartialOrd
                        + num_traits::Zero
                        + num_traits::One
                        + num_sqrt::Sqrt<Output = T>
                        + Copy,
                {
                    match self {
                        #(#call_variants,)*
                    }
                }
            }
        }
    }

    fn define_product_ops(self) -> impl Iterator<Item = syn::ItemImpl> {
        ProductOp::iter().map(move |op| {
            let op_ty = op.trait_ty();
            let op_fn = op.trait_fn();
            let variant_tuples = self.type_tuples().filter_map(|(lhs, rhs)| {
                op.output(self, lhs, rhs).map(|output| quote! {
                    (Value::#lhs(lhs), Value::#rhs(rhs)) => Some(Value::#output(#op_ty::#op_fn(lhs, rhs))),
                })
            });
            parse_quote! {
                impl<T> #op_ty<Self> for Value<T>
                where
                    T: std::ops::Mul<Output = T>
                        + std::ops::Add<Output = T>
                        + std::ops::Sub<Output = T>
                        + std::ops::Neg<Output = T>
                        + num_traits::Zero
                        + Copy,
                {
                    type Output = Option<Self>;
                    #[allow(unreachable_patterns)]
                    fn #op_fn(self, rhs: Self) -> Self::Output {
                        match (self, rhs) {
                            #(#variant_tuples)*
                            _ => None,
                        }
                    }
                }
            }
        })
    }

    fn define_sum_ops(self) -> impl Iterator<Item = syn::ItemImpl> {
        SumOp::iter().map(move |op| {
            let op_ty = op.trait_ty();
            let op_fn = op.trait_fn();
            let variant_tuples = self.type_tuples().filter_map(|(lhs, rhs)| {
                SumOp::sum(self, lhs, rhs).map(|output| quote! {
                    (Value::#lhs(lhs), Value::#rhs(rhs)) => Some(Value::#output(#op_ty::#op_fn(lhs, rhs))),
                })
            });
            parse_quote! {
                impl<T> #op_ty<Self> for Value<T>
                where
                    T: std::ops::Mul<Output = T>
                        + std::ops::Add<Output = T>
                        + std::ops::Sub<Output = T>
                        + std::ops::Neg<Output = T>
                        + num_traits::Zero
                        + Copy,
                {
                    type Output = Option<Self>;
                    fn #op_fn(self, rhs: Self) -> Self::Output {
                        match (self, rhs) {
                            #(#variant_tuples)*
                        }
                    }
                }
            }
        })
    }

    fn define_norm_ops(self) -> impl Iterator<Item = syn::ItemImpl> {
        NormOps::iter().map(move |op| {
            let op_ty = op.trait_ty();
            let op_fn = op.trait_fn();
            let variants = self.types().filter_map(|ty| {
                let scalar_blades = ty
                    .iter_blades_unsorted(self)
                    .map(|b| self.geo(b, b))
                    .any(|squared| !squared.is_zero());
                if scalar_blades {
                    Some(quote! {
                        Value::#ty(value) => Some(Value::Scalar(#op_ty::#op_fn(value))),
                    })
                } else {
                    None
                }
            });
            parse_quote! {
                impl<T> #op_ty for Value<T>
                where
                    T: std::ops::Mul<Output = T>
                        + std::ops::Add<Output = T>
                        + std::ops::Sub<Output = T>
                        + std::ops::Neg<Output = T>
                        + num_sqrt::Sqrt<Output = T>
                        + num_traits::Zero
                        + Copy,
                {
                    type Output = Option<Self>;
                    fn #op_fn(self) -> Self::Output {
                        match self {
                            #(#variants)*
                            _ => None,
                        }
                    }
                }
            }
        })
    }

    fn define_inv_ops(self) -> impl Iterator<Item = syn::ItemImpl> {
        InverseOps::iter().map(move |op| {
            let op_ty = op.trait_ty();
            let op_fn = op.trait_fn();
            let variants = self.types().filter_map(|ty| {
                if op.inapplicable(ty, self) {
                    None
                } else if op == InverseOps::Unitize {
                    Some(quote! {
                        Value::#ty(value) => {
                            if num_traits::Zero::is_zero(&value) {
                                None
                            } else {
                                Some(Value::#ty(#op_ty::#op_fn(value).value()))
                            }
                        },
                    })
                } else {
                    Some(quote! {
                        Value::#ty(value) => {
                            if num_traits::Zero::is_zero(&value) {
                                None
                            } else {
                                Some(Value::#ty(#op_ty::#op_fn(value)))
                            }
                        },
                    })
                }
            });
            parse_quote! {
                impl<T> #op_ty for Value<T>
                where
                    T: std::ops::Mul<Output = T>
                        + std::ops::Div<Output = T>
                        + std::ops::Add<Output = T>
                        + std::ops::Sub<Output = T>
                        + std::ops::Neg<Output = T>
                        + num_sqrt::Sqrt<Output = T>
                        + num_traits::Zero
                        + num_traits::One
                        + Copy,
                {
                    type Output = Option<Value<T>>;
                    fn #op_fn(self) -> Self::Output {
                        match self {
                            #(#variants)*
                            _ => None,
                        }
                    }
                }
            }
        })
    }

    fn define_sqrt(self) -> syn::ItemImpl {
        let trait_ty = Sqrt::trait_ty();
        let trait_fn = Sqrt::trait_fn();
        parse_quote! {
            impl<T> #trait_ty for Value<T>
            where
                T: num_sqrt::Sqrt<Output = T>
                    + PartialOrd
                    + num_traits::Zero,
            {
                type Output = Option<Self>;
                #[inline]
                fn #trait_fn(self) -> Self::Output {
                    match self {
                        Value::Scalar(value) => {
                            if value.s >= T::zero() {
                                Some(Value::Scalar(Scalar { s: value.s.sqrt() }))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                }
            }
        }
    }

    fn define_duals(self) -> impl Iterator<Item = syn::ItemImpl> {
        Complement::iter(self).map(move |op| {
            let trait_ty = op.trait_ty();
            let trait_fn = op.trait_fn();
            let variants = self.types().map(|ty| {
                let output = ty.complement(self);
                quote! {
                    Value::#ty(value) => Value::#output(#trait_ty::#trait_fn(value)),
                }
            });
            parse_quote! {
                impl<T> #trait_ty for Value<T>
                where
                    T: std::ops::Neg<Output = T>,
                {
                    type Output = Value<T>;
                    #[inline]
                    fn #trait_fn(self) -> Self::Output {
                        match self {
                            #(#variants)*
                        }
                    }
                }
            }
        })
    }

    fn define_rev(self) -> syn::ItemImpl {
        let rev_ty = Reverse::trait_ty();
        let rev_fn = Reverse::trait_fn();
        let variants = self.types().map(|ty| {
            quote! {
                Value::#ty(value) => Value::#ty(#rev_ty::#rev_fn(value)),
            }
        });
        parse_quote! {
            impl<T> #rev_ty for Value<T>
            where
                T: std::ops::Neg<Output = T>,
            {
                #[inline]
                fn #rev_fn(self) -> Self {
                    match self {
                        #(#variants)*
                    }
                }
            }
        }
    }

    fn define_neg(self) -> syn::ItemImpl {
        let neg_ty = quote!(std::ops::Neg);
        let neg_fn = quote!(neg);
        let variants = self.types().map(|ty| {
            quote! {
                Value::#ty(value) => Value::#ty(#neg_ty::#neg_fn(value)),
            }
        });
        parse_quote! {
            impl<T> #neg_ty for Value<T>
            where
                T: #neg_ty<Output = T>,
            {
                type Output = Value<T>;
                #[inline]
                fn #neg_fn(self) -> Self::Output {
                    match self {
                        #(#variants)*
                    }
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
enum BinaryOp {
    Add,
    Sub,
    Geo,
    Dot,
    Wedge,
    Antigeo,
    Antidot,
    Antiwedge,
    Grade(u32),
}

impl BinaryOp {
    pub fn iter(algebra: Algebra) -> impl Iterator<Item = Self> {
        use BinaryOp::*;
        let grade_products = (0..=algebra.bases.len())
            .into_iter()
            .map(|g| Grade(g as u32));
        IntoIterator::into_iter([Add, Sub, Geo, Dot, Wedge, Antigeo, Antidot, Antiwedge])
            .chain(grade_products)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum UnaryOp {
    Norm2,
    Norm,
    Unit,
    Inv,
    Sqrt,
    Dual,
    LeftComp,
    RightComp,
    Rev,
    Neg,
}

impl UnaryOp {
    pub fn iter(algebra: Algebra) -> impl Iterator<Item = Self> {
        use UnaryOp::*;
        if algebra.symmetrical_complements() {
            [Norm2, Norm, Unit, Inv, Sqrt, Dual, Rev, Neg]
                .as_slice()
                .iter()
                .copied()
        } else {
            [Norm2, Norm, Unit, Inv, Sqrt, LeftComp, RightComp, Rev, Neg]
                .as_slice()
                .iter()
                .copied()
        }
    }
}

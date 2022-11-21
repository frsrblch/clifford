use crate::algebra::{Complement, InverseOps, NormOps, ProductOp, SumOp, Type};
use crate::code_gen::{GradeProduct, Reverse, Sandwich, Sqrt};
use crate::Algebra;
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_quote, ItemEnum, ItemFn, ItemImpl};

impl Algebra {
    pub fn dynamic_types(self) -> TokenStream {
        let define_enum = self.define_enum();
        let impl_variant_fns = self.define_variant_fns();
        let binary_op = self.define_binary_op();
        let unary_op = self.define_unary_op();
        let product_ops = self.define_product_ops();
        let grade_products = self.impl_grade_products();
        let sandwich_ops = self.impl_sandwich_product();
        let sum_ops = self.define_sum_ops();
        let norm_ops = self.define_norm_ops();
        let inv_ops = self.define_inv_ops();
        let sqrt_ops = self.define_sqrt();
        let dual_ops = self.define_duals();
        let rev_ops = self.define_rev();
        let neg_ops = self.define_neg();
        let sin_cos = self.define_sin_cos();
        quote! {
            #define_enum
            #(#impl_variant_fns)*
            #binary_op
            #(#product_ops)*
            #(#grade_products)*
            #sandwich_ops
            #(#sum_ops)*
            #(#norm_ops)*
            #(#inv_ops)*
            #sqrt_ops
            #(#dual_ops)*
            #rev_ops
            #neg_ops
            #unary_op
            #(#sin_cos)*
        }
    }

    fn define_enum(self) -> ItemEnum {
        let variants = self.types().map(|ty| {
            quote! {
                #ty(#ty<T>),
            }
        });
        parse_quote! {
            #[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
            pub enum Value<T> {
                #(#variants)*
            }
        }
    }

    fn define_binary_op(self) -> TokenStream {
        let variants = BinaryOp::iter(self);
        let variants1 = BinaryOp::iter(self);
        let len = proc_macro2::Literal::usize_unsuffixed(BinaryOp::iter(self).count());
        let call_variants = BinaryOp::iter(self).map(|op| match op {
            BinaryOp::Add => quote! { BinaryOp::Add => std::ops::Add::add(lhs, rhs), },
            BinaryOp::Sub => quote! { BinaryOp::Sub => std::ops::Sub::sub(lhs, rhs), },
            BinaryOp::Geo => quote! { BinaryOp::Geo => Geo::geo(lhs, rhs), },
            BinaryOp::Dot => quote! { BinaryOp::Dot => Dot::dot(lhs, rhs), },
            BinaryOp::Wedge => quote! { BinaryOp::Wedge => Wedge::wedge(lhs, rhs), },
            BinaryOp::Sandwich => {
                quote! { BinaryOp::Sandwich => Sandwich::sandwich(lhs, rhs), }
            }
            BinaryOp::Grade(g) => {
                let grade = Type::Grade(g);
                quote! { BinaryOp::#grade => #grade::product(lhs, rhs), }
            }
        });
        quote! {
            #[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
            pub enum BinaryOp {
                #(#variants,)*
            }

            impl BinaryOp {
                pub fn values() -> [Self; #len] {
                    use BinaryOp::*;
                    [
                        #(#variants1,)*
                    ]
                }

                pub fn len() -> usize {
                    Self::values().len()
                }

                pub fn call<T>(self, lhs: Value<T>, rhs: Value<T>) -> Option<Value<T>>
                where
                    T: std::ops::Mul<Output = T>
                        + std::ops::Div<Output = T>
                        + std::ops::Add<Output = T>
                        + std::ops::Sub<Output = T>
                        + std::ops::Neg<Output = T>
                        + num_traits::One
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

    fn impl_sandwich_product(self) -> ItemImpl {
        let trait_ty = Sandwich::trait_ty();
        let trait_fn = Sandwich::trait_fn();

        let arms = self.type_tuples().filter_map(|(lhs, rhs)| {
            if ProductOp::Geo.output(self, lhs, rhs).is_none()
                || !matches!(rhs, Type::Grade(_))
                || InverseOps::Inverse.inapplicable(lhs, self)
            {
                None
            } else {
                Some(quote! {
                    (Value::#lhs(lhs), Value::#rhs(rhs)) => Some(Value::#rhs(lhs.sandwich(rhs))),
                })
            }
        });

        parse_quote! {
            impl<T> #trait_ty<Value<T>> for Value<T>
            where
                T: std::ops::Mul<Output = T>
                    + std::ops::Div<Output = T>
                    + std::ops::Add<Output = T>
                    + std::ops::Sub<Output = T>
                    + std::ops::Neg<Output = T>
                    + num_traits::Zero
                    + num_traits::One
                    + Copy,
            {
                type Output = Option<Value<T>>;
                #[inline]
                fn #trait_fn(self, rhs: Value<T>) -> Self::Output {
                    match (self, rhs) {
                        #(#arms)*
                        _ => None,
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
                if GradeProduct::contains(lhs, rhs, grade, self) && ProductOp::Geo.output(self, lhs, rhs) != Some(grade) {
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
        let variants = UnaryOp::iter(self);
        let values = UnaryOp::iter(self).map(|op| quote!(Self::#op));
        let len = proc_macro2::Literal::usize_unsuffixed(UnaryOp::iter(self).count());

        let call_variants = UnaryOp::iter(self).map(|op| match op {
            UnaryOp::Norm2 => quote! { UnaryOp::Norm2 => value.norm2() },
            UnaryOp::Norm => quote! { UnaryOp::Norm => value.norm() },
            UnaryOp::Unit => quote! { UnaryOp::Unit => value.unit() },
            UnaryOp::Inv => quote! { UnaryOp::Inv => value.inv() },
            UnaryOp::Sqrt => quote! { UnaryOp::Sqrt => num_sqrt::Sqrt::sqrt(value) },
            UnaryOp::Dual => quote! { UnaryOp::Dual => Some(value.dual()) },
            UnaryOp::LeftComp => quote! { UnaryOp::LeftComp => Some(value.left_comp()) },
            UnaryOp::RightComp => quote! { UnaryOp::RightComp => Some(value.right_comp()) },
            UnaryOp::Rev => quote! { UnaryOp::Rev => value.rev() },
            UnaryOp::Neg => quote! { UnaryOp::Neg => Some(-value) },
            UnaryOp::Sin => quote! { UnaryOp::Sin => num_trig::Sin::sin(value) },
            UnaryOp::Cos => quote! { UnaryOp::Cos => num_trig::Cos::cos(value) },
        });

        quote! {
            #[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
            pub enum UnaryOp {
                #(#variants,)*
            }

            impl UnaryOp {
                pub fn values() -> [Self; #len] {
                    [
                        #(#values,)*
                    ]
                }

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
                        + num_trig::Sin<Output = T>
                        + num_trig::Cos<Output = T>
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
        ProductOp::iter_all(self).map(move |op| {
            let op_ty = op.trait_ty();
            let op_fn = op.trait_fn();
            let variant_tuples = self.type_tuples().filter_map(|(lhs, rhs)| {
                let output = op.output(self, lhs, rhs)?;
                // skip redundant ops (scalar dot scalar, etc)
                // if matches!(op, ProductOp::Dot | ProductOp::Wedge) && ProductOp::Geo.output(self, lhs, rhs) == Some(output) {
                //     return None;
                // }
                Some(quote! {
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
                if ty == Type::Grade(0) && op == NormOps::Norm {
                    return None;
                }

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
                    #[inline]
                    #[allow(unreachable_patterns)]
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
                    if ty == Type::Grade(0) {
                        None
                    } else {
                        Some(quote! {
                            Value::#ty(value) => {
                                if num_traits::Zero::is_zero(&value) {
                                    None
                                } else {
                                    Some(Value::#ty(#op_ty::#op_fn(value).value()))
                                }
                            },
                        })
                    }
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
                Scalar<T>: num_sqrt::Sqrt<Output = Scalar<T>>,
                Bivector<T>: num_sqrt::Sqrt<Output = Motor<T>>
                    + num_traits::Zero,
                Motor<T>: num_sqrt::Sqrt<Output = Motor<T>>
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
                        Value::Bivector(value) => {
                            if num_traits::Zero::is_zero(&value) {
                                None
                            } else {
                                Some(
                                    Value::Motor(
                                        num_sqrt::Sqrt::sqrt(value)
                                    )
                                )
                            }
                        }
                        Value::Motor(value) => {
                            if num_traits::Zero::is_zero(&value) {
                                None
                            } else {
                                Some(
                                    Value::Motor(
                                        num_sqrt::Sqrt::sqrt(value)
                                    )
                                )
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
        let variants = self.types().filter_map(|ty| {
            if matches!(ty, Type::Grade(0) | Type::Grade(1)) {
                return None;
            }
            Some(quote! {
                Value::#ty(value) => Some(Value::#ty(#rev_ty::#rev_fn(value))),
            })
        });
        parse_quote! {
            impl<T> #rev_ty for Value<T>
            where
                T: std::ops::Neg<Output = T>,
            {
                type Output = Option<Self>;
                #[inline]
                fn #rev_fn(self) -> Self::Output {
                    match self {
                        #(#variants)*
                        _ => None,
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

    fn define_sin_cos(self) -> [syn::ItemImpl; 2] {
        [
            parse_quote! {
                impl<T> num_trig::Sin for Value<T>
                where
                    T: num_trig::Sin<Output = T>
                {
                    type Output = Option<Self>;
                    #[inline]
                    fn sin(self) -> Self::Output {
                        match self {
                            Value::Scalar(Scalar { s }) => Some(Value::Scalar(Scalar { s: num_trig::Sin::sin(s) })),
                            _ => None,
                        }
                    }
                }
            },
            parse_quote! {
                impl<T> num_trig::Cos for Value<T>
                where
                    T: num_trig::Cos<Output = T>
                {
                    type Output = Option<Self>;
                    #[inline]
                    fn cos(self) -> Self::Output {
                        match self {
                            Value::Scalar(Scalar { s }) => Some(Value::Scalar(Scalar { s: num_trig::Cos::cos(s) })),
                            _ => None,
                        }
                    }
                }
            },
        ]
    }

    fn define_variant_fns(&self) -> [ItemImpl; 2] {
        let fns = self.types().map(|ty| -> ItemFn {
            let ident = ty.fn_ident();
            parse_quote! {
                pub fn #ident() -> Value<f_zero::f0> {
                    Value::#ty(#ty::default())
                }
            }
        });

        let vars = self.types().map(|ty| {
            let ident = ty.fn_ident();
            quote! {
                Value::#ty(_) => Value::#ident(),
            }
        });

        [
            parse_quote! {
                impl Value<f_zero::f0> {
                    #(#fns)*
                }
            },
            parse_quote! {
                impl<T> Value<T> {
                    pub fn ty(&self) -> Value<f_zero::f0> {
                        match self {
                            #(#vars)*
                        }
                    }
                }
            },
        ]
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
enum BinaryOp {
    Add,
    Sub,
    Geo,
    Dot,
    Wedge,
    Sandwich,
    Grade(u32),
}

impl BinaryOp {
    pub fn iter(algebra: Algebra) -> impl Iterator<Item = Self> {
        use BinaryOp::*;
        let grade_products = (0..=algebra.bases.len())
            .into_iter()
            .map(|g| Grade(g as u32));
        IntoIterator::into_iter([Add, Sub, Geo, Dot, Wedge, Sandwich]).chain(grade_products)
    }
}

impl ToTokens for BinaryOp {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Add => quote!(Add).to_tokens(tokens),
            Self::Sub => quote!(Sub).to_tokens(tokens),
            Self::Geo => ProductOp::Geo.trait_ty().to_tokens(tokens),
            Self::Dot => ProductOp::Dot.trait_ty().to_tokens(tokens),
            Self::Wedge => ProductOp::Wedge.trait_ty().to_tokens(tokens),
            Self::Sandwich => Sandwich::trait_ty().to_tokens(tokens),
            Self::Grade(g) => Type::Grade(*g).to_tokens(tokens),
        }
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
    Sin,
    Cos,
}

impl UnaryOp {
    pub fn iter(algebra: Algebra) -> impl Iterator<Item = Self> {
        use UnaryOp::*;
        if algebra.symmetrical_complements() {
            vec![Norm2, Norm, Unit, Inv, Sqrt, Dual, Rev, Neg, Sin, Cos].into_iter()
        } else {
            vec![
                Norm2, Norm, Unit, Inv, Sqrt, LeftComp, RightComp, Rev, Neg, Sin, Cos,
            ]
            .into_iter()
        }
    }
}

impl ToTokens for UnaryOp {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let t = match self {
            Self::Norm2 => quote!(Norm2),
            Self::Norm => quote!(Norm),
            Self::Unit => quote!(Unit),
            Self::Inv => quote!(Inv),
            Self::Sqrt => quote!(Sqrt),
            Self::Dual => quote!(Dual),
            Self::LeftComp => quote!(LeftComp),
            Self::RightComp => quote!(RightComp),
            Self::Rev => quote!(Rev),
            Self::Neg => quote!(Neg),
            Self::Sin => quote!(Sin),
            Self::Cos => quote!(Cos),
        };
        t.to_tokens(tokens);
    }
}

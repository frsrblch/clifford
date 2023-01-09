use crate::algebra::*;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use std::collections::HashMap;

impl Algebra {
    pub fn define(&self) -> TokenStream {
        let structs = self.types().map(move |ty| ty.define(self));

        let scalar_num_traits = impl_num_traits_for_scalar();
        let scalar_float = impl_float_for_scalar(self);

        let impl_new_fn = self.types().map(|ty| ty.impl_new_fn(self));
        let impl_map_fn = self.types().map(|ty| ty.impl_map(self));
        let impl_grade_fns = self.types().filter_map(|ty| ty.impl_grade_fns(self));
        let impl_from = self.types().flat_map(|ty| ty.impl_from(self));

        let impl_zero = self.types().map(|ty| ty.impl_zero(self));
        let impl_one = self.types().filter_map(|ty| ty.impl_one(self));
        let impl_bytemuck = self.types().map(ImplBytemuck);
        let impl_float_type = self.types().map(ImplFloatType);

        let impl_product_ops = self.type_tuples().flat_map(|(lhs, rhs)| {
            ProductOp::iter_all(self).filter_map(move |op| op.impl_for(self, lhs, rhs))
        });

        let operator_overloads = self.type_tuples().flat_map(|(lhs, rhs)| {
            Overload::iter(self).filter_map(move |op| op.impl_for(lhs, rhs, self))
        });

        let unary_operator_overloads = UnaryOverload::iter(self)
            .flat_map(|op| self.types().map(move |ty| op.impl_for(ty, self)));

        let div_ops = self
            .type_tuples()
            .filter_map(|(lhs, rhs)| Div::impl_for(lhs, rhs, self));

        let impl_sum_ops = SumOp::iter().flat_map(|op| {
            self.type_tuples()
                .filter_map(move |(lhs, rhs)| op.impl_for(self, lhs, rhs))
        });

        let impl_neg = self.types().map(|ty| Neg::impl_for(ty, self));

        let impl_rev = self.types().map(|ty| Reverse::impl_for(ty, self));

        let impl_complements = Complement::iter(self)
            .flat_map(|comp| self.types().map(move |ty| ty.impl_complement(self, comp)));

        let explicit_scalar_ops = ScalarOps::iter().flat_map(|op| {
            self.types()
                .flat_map(move |ty| op.impl_for_scalar(ty, self))
        });

        let scalar_assign_ops = ScalarAssignOps::iter()
            .flat_map(|op| self.types().map(move |ty| op.impl_for(ty, self)));

        let sum_assign_ops =
            SumAssignOps::iter().flat_map(|op| self.types().map(move |ty| op.impl_for(ty, self)));

        let float_conversion = FloatConversion::iter()
            .flat_map(|op| self.types().map(move |ty| op.impl_for(ty, self)));

        let grade_products = self.grades().flat_map(|out| {
            self.type_tuples()
                .map(move |(lhs, rhs)| GradeProduct::impl_for(lhs, rhs, out, self))
        });

        let norm_ops =
            NormOps::iter().flat_map(|op| self.types().map(move |ty| op.impl_for(ty, self)));

        let sandwich_ops = self
            .type_tuples()
            .map(|(lhs, rhs)| Sandwich::impl_for(lhs, rhs, self));

        let antisandwich_ops = self
            .type_tuples()
            .map(|(lhs, rhs)| Antisandwich::impl_for(lhs, rhs, self));

        let unit_items = Unit::define(self);

        let sample_unit = self.types().filter_map(|ty| ty.impl_sample_unit(self));

        let inverse_ops = InverseOps::iter()
            .flat_map(|op| self.types().filter_map(move |ty| op.impl_for(ty, self)));

        let test_unit_inv =
            InverseOps::iter().flat_map(|op| self.types().filter_map(move |ty| op.tests(ty, self)));

        let test_sample_unit = self.types().map(|ty| ty.test_sample_unit(self));

        quote!(
            #(#structs)*
            #scalar_num_traits
            #scalar_float
            #(#impl_new_fn)*
            #(#impl_map_fn)*
            #(#impl_grade_fns)*
            #(#impl_from)*
            #(#impl_zero)*
            #(#impl_one)*
            #(#impl_bytemuck)*
            #(#impl_float_type)*
            #(#impl_product_ops)*
            #(#operator_overloads)*
            #(#unary_operator_overloads)*
            #(#div_ops)*
            #(#impl_sum_ops)*
            #(#impl_neg)*
            #(#impl_rev)*
            #(#impl_complements)*
            #(#explicit_scalar_ops)*
            #(#scalar_assign_ops)*
            #(#sum_assign_ops)*
            #(#float_conversion)*
            #(#grade_products)*
            #(#norm_ops)*
            #(#sandwich_ops)*
            #(#antisandwich_ops)*
            #(#inverse_ops)*
            #(#unit_items)*
            #(#sample_unit)*

            #[cfg(test)]
            mod tests {
                #[allow(unused_imports)]
                use super::*;
                #(#test_unit_inv)*
                #(#test_sample_unit)*
            }
        )
    }
}

fn impl_num_traits_for_scalar() -> TokenStream {
    quote! {
        macro_rules! impl_to_primitive {
            ( $($fn_ident:ident, $T:ty,)* ) => {
                impl<T> num_traits::ToPrimitive for Scalar<T>
                where
                    T: num_traits::ToPrimitive,
                {
                    $(
                        fn $fn_ident(&self) -> Option<$T> {
                            self.s.$fn_ident()
                        }
                    )*
                }
            };
        }

        impl_to_primitive! {
            to_i64, i64,
            to_u64, u64,
            to_isize, isize,
            to_usize, usize,
            to_i8, i8,
            to_u8, u8,
            to_i16, i16,
            to_u16, u16,
            to_i32, i32,
            to_u32, u32,
            to_u128, u128,
            to_i128, i128,
            to_f32, f32,
            to_f64, f64,
        }

        macro_rules! impl_from_primitive {
            ( $($fn_ident:ident, $T:ty,)* ) => {
                impl<T> num_traits::FromPrimitive for Scalar<T>
                where
                    T: num_traits::FromPrimitive,
                {
                    $(
                        fn $fn_ident(t: $T) -> Option<Self> {
                            T::$fn_ident(t).map(|s| Scalar { s })
                        }
                    )*
                }
            };
        }

        impl_from_primitive! {
            from_i64, i64,
            from_u64, u64,
            from_isize, isize,
            from_usize, usize,
            from_i8, i8,
            from_u8, u8,
            from_i16, i16,
            from_u16, u16,
            from_i32, i32,
            from_u32, u32,
            from_u128, u128,
            from_i128, i128,
            from_f32, f32,
            from_f64, f64,
        }

        macro_rules! impl_float_const_for_scalar {
            ($($fn_:ident,)*) => {
                impl<T> num_traits::FloatConst for Scalar<T> where T: num_traits::FloatConst {
                    $(
                        fn $fn_() -> Self {
                            Scalar { s: T::$fn_() }
                        }
                    )*
                }
            };
        }

        impl_float_const_for_scalar! {
            E,
            FRAC_1_PI,
            FRAC_1_SQRT_2,
            FRAC_2_PI,
            FRAC_2_SQRT_PI,
            FRAC_PI_2,
            FRAC_PI_3,
            FRAC_PI_4,
            FRAC_PI_6,
            FRAC_PI_8,
            LN_10,
            LN_2,
            LOG10_E,
            LOG2_E,
            PI,
            SQRT_2,
        }

        impl<T> num_traits::NumCast for Scalar<T>
        where
            T: num_traits::NumCast + num_traits::ToPrimitive,
        {
            fn from<N: num_traits::ToPrimitive>(n: N) -> Option<Self> {
                T::from(n).map(|s| Scalar { s })
            }
        }

        impl<T> std::ops::Rem for Scalar<T>
        where
            T: std::ops::Rem<Output = T>,
        {
            type Output = Scalar<T>;
            #[inline]
            fn rem(self, rhs: Self) -> Self::Output {
                Scalar {
                    s: std::ops::Rem::rem(self.s, rhs.s),
                }
            }
        }

        impl<T: PartialEq> PartialEq<T> for Scalar<T> {
            fn eq(&self, other: &T) -> bool {
                self.s.eq(other)
            }
        }

        impl PartialEq<Scalar<f32>> for f32 {
            fn eq(&self, other: &Scalar<f32>) -> bool {
                self.eq(&other.s)
            }
        }

        impl PartialEq<Scalar<f64>> for f64 {
            fn eq(&self, other: &Scalar<f64>) -> bool {
                self.eq(&other.s)
            }
        }

        impl<T: PartialOrd> PartialOrd<T> for Scalar<T> {
            fn partial_cmp(&self, rhs: &T) -> Option<std::cmp::Ordering> {
                self.s.partial_cmp(rhs)
            }
        }

        impl PartialOrd<Scalar<f32>> for f32 {
            fn partial_cmp(&self, rhs: &Scalar<f32>) -> Option<std::cmp::Ordering> {
                self.partial_cmp(&rhs.s)
            }
        }

        impl PartialOrd<Scalar<f64>> for f64 {
            fn partial_cmp(&self, rhs: &Scalar<f64>) -> Option<std::cmp::Ordering> {
                self.partial_cmp(&rhs.s)
            }
        }
    }
}

fn impl_float_for_scalar(algebra: &Algebra) -> TokenStream {
    if InverseOps::Inverse.inapplicable(Type::Grade(0), algebra) {
        return quote!();
    }

    quote! {
        impl<T> num_traits::Num for Scalar<T>
        where
            T: num_traits::Float,
        {
            type FromStrRadixErr = T::FromStrRadixErr;
            #[inline]
            fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                T::from_str_radix(str, radix).map(|s| Scalar { s })
            }
        }

        impl<T> num_traits::Float for Scalar<T>
        where
            T: num_traits::Float,
        {
            #[inline]
            fn nan() -> Self {
                Scalar { s: T::nan() }
            }

            #[inline]
            fn infinity() -> Self {
                Scalar { s: T::infinity() }
            }

            #[inline]
            fn neg_infinity() -> Self {
                Scalar { s: T::neg_infinity() }
            }

            #[inline]
            fn neg_zero() -> Self {
                Scalar { s: T::neg_zero() }
            }

            #[inline]
            fn min_value() -> Self {
                Scalar { s: T::min_value() }
            }

            #[inline]
            fn min_positive_value() -> Self {
                Scalar { s: T::min_positive_value() }
            }

            #[inline]
            fn max_value() -> Self {
                Scalar { s: T::max_value() }
            }

            #[inline]
            fn is_nan(self) -> bool {
                self.s.is_nan()
            }

            #[inline]
            fn is_infinite(self) -> bool {
                self.s.is_infinite()
            }

            #[inline]
            fn is_finite(self) -> bool {
                self.s.is_finite()
            }

            #[inline]
            fn is_normal(self) -> bool {
                self.s.is_normal()
            }

            #[inline]
            fn classify(self) -> std::num::FpCategory {
                self.s.classify()
            }

            #[inline]
            fn floor(self) -> Self {
                self.map(num_traits::Float::floor)
            }

            #[inline]
            fn ceil(self) -> Self {
                self.map(num_traits::Float::ceil)
            }

            #[inline]
            fn round(self) -> Self {
                self.map(num_traits::Float::round)
            }

            #[inline]
            fn trunc(self) -> Self {
                self.map(num_traits::Float::trunc)
            }

            #[inline]
            fn fract(self) -> Self {
                self.map(num_traits::Float::fract)
            }

            #[inline]
            fn abs(self) -> Self {
                self.map(num_traits::Float::abs)
            }

            #[inline]
            fn signum(self) -> Self {
                self.map(num_traits::Float::signum)
            }

            #[inline]
            fn is_sign_positive(self) -> bool {
                self.s.is_sign_positive()
            }

            #[inline]
            fn is_sign_negative(self) -> bool {
                self.s.is_sign_negative()
            }

            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                Scalar { s: self.s.mul_add(a.s, b.s) }
            }

            #[inline]
            fn recip(self) -> Self {
                Scalar { s: self.s.recip() }
            }

            #[inline]
            fn powi(self, n: i32) -> Self {
                Scalar { s: self.s.powi(n) }
            }

            #[inline]
            fn powf(self, n: Self) -> Self {
                Scalar { s: self.s.powf(n.s) }
            }

            #[inline]
            fn sqrt(self) -> Self {
                self.map(num_traits::Float::sqrt)
            }

            #[inline]
            fn exp(self) -> Self {
                self.map(num_traits::Float::exp)
            }

            #[inline]
            fn exp2(self) -> Self {
                self.map(num_traits::Float::exp2)
            }

            #[inline]
            fn ln(self) -> Self {
                self.map(num_traits::Float::ln)
            }

            #[inline]
            fn log(self, base: Self) -> Self {
                Scalar { s: self.s.log(base.s) }
            }

            #[inline]
            fn log2(self) -> Self {
                self.map(num_traits::Float::log2)
            }

            #[inline]
            fn log10(self) -> Self {
                self.map(num_traits::Float::log10)
            }

            #[inline]
            fn max(self, other: Self) -> Self {
                Scalar { s: self.s.max(other.s) }
            }

            #[inline]
            fn min(self, other: Self) -> Self {
                Scalar { s: self.s.min(other.s) }
            }

            #[inline]
            fn abs_sub(self, other: Self) -> Self {
                Scalar { s: self.s.abs_sub(other.s) }
            }

            #[inline]
            fn cbrt(self) -> Self {
                self.map(num_traits::Float::cbrt)
            }

            #[inline]
            fn hypot(self, other: Self) -> Self {
                Scalar { s: self.s.hypot(other.s) }
            }

            #[inline]
            fn sin(self) -> Self {
                self.map(num_traits::Float::sin)
            }

            #[inline]
            fn cos(self) -> Self {
                self.map(num_traits::Float::cos)
            }

            #[inline]
            fn tan(self) -> Self {
                self.map(num_traits::Float::tan)
            }

            #[inline]
            fn asin(self) -> Self {
                self.map(num_traits::Float::asin)
            }

            #[inline]
            fn acos(self) -> Self {
                self.map(num_traits::Float::acos)
            }

            #[inline]
            fn atan(self) -> Self {
                self.map(num_traits::Float::atan)
            }

            #[inline]
            fn atan2(self, other: Self) -> Self {
                Scalar { s: self.s.atan2(other.s) }
            }

            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                let (sin, cos) = self.s.sin_cos();
                (Scalar { s: sin }, Scalar { s: cos })
            }

            #[inline]
            fn exp_m1(self) -> Self {
                self.map(num_traits::Float::exp_m1)
            }

            #[inline]
            fn ln_1p(self) -> Self {
                self.map(num_traits::Float::ln_1p)
            }

            #[inline]
            fn sinh(self) -> Self {
                self.map(num_traits::Float::sinh)
            }

            #[inline]
            fn cosh(self) -> Self {
                self.map(num_traits::Float::cosh)
            }

            #[inline]
            fn tanh(self) -> Self {
                self.map(num_traits::Float::tanh)
            }

            #[inline]
            fn asinh(self) -> Self {
                self.map(num_traits::Float::asinh)
            }

            #[inline]
            fn acosh(self) -> Self {
                self.map(num_traits::Float::acosh)
            }

            #[inline]
            fn atanh(self) -> Self {
                self.map(num_traits::Float::atanh)
            }

            #[inline]
            fn integer_decode(self) -> (u64, i16, i8) {
                self.s.integer_decode()
            }

            #[inline]
            fn epsilon() -> Self {
                Scalar { s: T::epsilon() }
            }

            #[inline]
            fn to_degrees(self) -> Self {
                self.map(num_traits::Float::to_degrees)
            }

            #[inline]
            fn to_radians(self) -> Self {
                self.map(num_traits::Float::to_radians)
            }

            #[inline]
            fn copysign(self, sign: Self) -> Self {
                Scalar { s: self.s.copysign(sign.s) }
            }
        }
    }
}

fn fn_attrs() -> TokenStream {
    quote!(
        #[inline]
        #[track_caller]
    )
}

struct ImplFloatType(Type);

impl ToTokens for ImplFloatType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let ty = self.0;
        tokens.extend(quote! {
            impl<T> geo_traits::FloatType for #ty<T>
            where
                T: num_traits::Float,
            {
                type Float = T;
            }
        });
    }
}

struct ImplBytemuck(Type);

impl ToTokens for ImplBytemuck {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let ty = self.0;
        tokens.extend(quote! {
            unsafe impl<T: bytemuck::Zeroable> bytemuck::Zeroable for #ty<T> {}
            unsafe impl<T: bytemuck::Pod + bytemuck::Zeroable> bytemuck::Pod for #ty<T> {}
        });
    }
}

pub struct Neg;

impl Neg {
    fn impl_for(ty: Type, _algebra: &Algebra) -> TokenStream {
        let fn_attrs = fn_attrs();
        quote! {
            impl<T> std::ops::Neg for #ty<T>
            where
                T: num_traits::Float,
            {
                type Output = #ty<T>;
                #fn_attrs
                fn neg(self) -> Self::Output {
                    self.map(std::ops::Neg::neg)
                }
            }
        }
    }
}

pub struct Reverse;

impl Reverse {
    fn impl_for(ty: Type, algebra: &Algebra) -> TokenStream {
        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();
        let fn_attrs = fn_attrs();

        let fields = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field_ref(algebra);
            let rev = blade.rev();
            if rev.is_positive() {
                quote! { #f: self.#f, }
            } else {
                quote! { #f: -self.#f, }
            }
        });

        quote! {
            impl<T> #trait_ty for #ty<T> where T: num_traits::Float {
                type Output = #ty<T>;
                #fn_attrs
                fn #trait_fn(self) -> Self {
                    #ty {
                        #(#fields)*
                    }
                }
            }
        }
    }

    pub fn trait_ty() -> TokenStream {
        quote!(geo_traits::Reverse)
    }

    pub fn trait_fn() -> TokenStream {
        quote!(rev)
    }
}

pub struct GradeProduct;

impl GradeProduct {
    pub fn has_implementation(lhs: Type, rhs: Type, out: Type, algebra: &Algebra) -> bool {
        if algebra.slim && !out.is_scalar() {
            return false;
        }

        let blades = Self::iter_blades(lhs, rhs, out, algebra).fold(
            HashMap::<Blade, Vec<(Blade, Blade, Blade)>>::new(),
            |mut map, (l, r, o)| {
                map.entry(o.unsigned()).or_default().push((l, r, o));
                map
            },
        );

        if blades.is_empty() {
            return false;
        }

        true
    }

    pub fn impl_for(lhs: Type, rhs: Type, out: Type, algebra: &Algebra) -> Option<TokenStream> {
        if !Self::has_implementation(lhs, rhs, out, algebra) {
            return None;
        }

        let blades = Self::iter_blades(lhs, rhs, out, algebra).fold(
            HashMap::<Blade, Vec<(Blade, Blade, Blade)>>::new(),
            |mut map, (l, r, o)| {
                map.entry(o.unsigned()).or_default().push((l, r, o));
                map
            },
        );

        let fn_attrs = fn_attrs();
        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();

        let fields = out.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field_ref(algebra);
            let mut sum = quote!();

            if let Some(vec) = blades.get(&blade) {
                for (l, r, o) in vec {
                    let lf = l.field_ref(algebra);
                    let rf = r.field_ref(algebra);
                    match (sum.is_empty(), o.is_positive()) {
                        (true, true) => sum.extend(quote! { lhs.#lf * rhs.#rf }),
                        (true, false) => sum.extend(quote! { -(lhs.#lf * rhs.#rf) }),
                        (false, true) => sum.extend(quote! { + lhs.#lf * rhs.#rf }),
                        (false, false) => sum.extend(quote! { - lhs.#lf * rhs.#rf }),
                    }
                }
            } else {
                sum = quote! { num_traits::Zero::zero() };
            }

            quote! {
                #f: #sum,
            }
        });

        Some(quote! {
            impl<T, U, V> #trait_ty<#lhs<T>, #rhs<U>> for #out<V>
            where
                T: num_traits::Float + std::ops::Mul<U, Output = V>,
                U: num_traits::Float,
                V: num_traits::Float,
            {
                type Output = #out<V>;
                #fn_attrs
                fn #trait_fn(lhs: #lhs<T>, rhs: #rhs<U>) -> Self::Output {
                    #out {
                        #(#fields)*
                    }
                }
            }
        })
    }

    pub fn trait_ty() -> TokenStream {
        quote!(geo_traits::GradeProduct)
    }

    pub fn trait_fn() -> TokenStream {
        quote!(product)
    }

    fn iter_blades(
        lhs: Type,
        rhs: Type,
        out: Type,
        algebra: &Algebra,
    ) -> impl Iterator<Item = (Blade, Blade, Blade)> + '_ {
        lhs.iter_blades_unsorted(algebra)
            .flat_map(move |lhs| {
                rhs.iter_blades_unsorted(algebra)
                    .map(move |rhs| (lhs, rhs, algebra.geo(lhs, rhs)))
            })
            .filter(move |(_l, _r, o)| out.contains(*o))
    }
}

impl Type {
    pub fn define(self, algebra: &Algebra) -> TokenStream {
        let fields = self
            .iter_blades_sorted(algebra)
            .map(|b| b.field_ref(algebra));

        let attr = if self.single_blade(algebra) {
            quote! {
                #[repr(C)]
                #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
            }
        } else {
            quote! {
                #[repr(C)]
                #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
            }
        };

        quote! {
            #attr
            pub struct #self <T> {
                #( pub #fields: T, )*
            }
        }
    }

    fn impl_grade_fns(self, algebra: &Algebra) -> Option<TokenStream> {
        let contained_types = || {
            algebra
                .types()
                .filter(|&ty| ty != self && self.contains_ty(ty))
        };

        if !contained_types().any(|_| true) {
            return None;
        }

        let fns = contained_types().map(|rhs| {
            let grade_fn = rhs.fn_ident();
            let fields = rhs.iter_blades_unsorted(algebra).map(|blade| {
                let f = blade.field_ref(algebra);
                quote!( #f: self.#f, )
            });
            quote! {
                pub fn #grade_fn(self) -> #rhs<T> {
                    #rhs {
                        #(#fields)*
                    }
                }
            }
        });
        Some(quote! {
            impl<T> #self<T> {
                #(#fns)*
            }
        })
    }

    fn impl_new_fn(self, algebra: &Algebra) -> TokenStream {
        let params = self.iter_blades_sorted(algebra).map(|blade| {
            let f = blade.field_ref(algebra);
            quote!(#f: T)
        });

        let fields = self.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field_ref(algebra);
            quote! {
                #f: {
                    debug_assert!(#f.is_finite());
                    #f
                }
            }
        });

        quote! {
            impl<T: num_traits::Float> #self<T> {
                /// A constructor that asserts that the inputs are finite.
                #[inline]
                #[allow(clippy::too_many_arguments)]
                pub fn new(#(#params),*) -> #self<T> {
                    #self {
                        #(#fields),*
                    }
                }
            }
        }
    }

    pub fn impl_map(self, algebra: &Algebra) -> TokenStream {
        let fields = self.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field_ref(algebra);
            quote! {
                #f: f(self.#f),
            }
        });
        quote! {
            impl<T> #self<T> {
                /// Takes a closure and calls that closure on each element, returning a new value
                pub fn map<U, F: Fn(T) -> U>(self, f: F) -> #self<U> {
                    #self {
                        #(#fields)*
                    }
                }
            }
        }
    }

    pub fn impl_from(self, algebra: &Algebra) -> impl Iterator<Item = TokenStream> + '_ {
        algebra.types().filter_map(move |target| {
            if self != target && target.contains_ty(self) {
                let fields = self.iter_blades_unsorted(algebra).map(|blade| {
                    let f = blade.field_ref(algebra);
                    quote!(#f: value.#f,)
                });
                Some(quote! {
                    impl<T> From<#self<T>> for #target<T>
                    where
                        #target<T>: num_traits::Zero,
                    {
                        fn from(value: #self<T>) -> #target<T> {
                            #target {
                                #(#fields)*
                                ..num_traits::Zero::zero()
                            }
                        }
                    }
                })
            } else {
                None
            }
        })
    }

    fn ident(self) -> Ident {
        Ident::new(self.name(), Span::mixed_site())
    }

    fn impl_complement(self, algebra: &Algebra, op: Complement) -> TokenStream {
        let ident = op.trait_ty();
        let fn_ident = op.trait_fn();
        let comp = self.complement(algebra);
        let fn_attrs = fn_attrs();
        let fields = self.iter_blades_unsorted(algebra).map(|blade| {
            let comp = op.call(algebra, blade);
            let cf = comp.field_ref(algebra);
            let sf = blade.field_ref(algebra);
            if comp.is_positive() {
                quote!(#cf: self.#sf,)
            } else {
                quote!(#cf: -self.#sf,)
            }
        });
        quote! {
            impl<T> #ident for #self<T>
            where
                T: num_traits::Float,
            {
                type Output = #comp<T>;
                #fn_attrs
                fn #fn_ident(self) -> Self::Output {
                    #comp {
                        #(#fields)*
                    }
                }
            }
        }
    }

    fn impl_zero(self, algebra: &Algebra) -> TokenStream {
        let fields0 = self.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field_ref(algebra);
            quote! { #f: num_traits::Zero::zero(), }
        });
        let fields1 = self.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field_ref(algebra);
            quote! { self.#f.is_zero() }
        });
        quote! {
            impl<T> num_traits::Zero for #self<T>
            where
                T: num_traits::Float,
            {
                #[inline]
                fn zero() -> Self {
                    #self {
                        #(#fields0)*
                    }
                }

                #[inline]
                fn is_zero(&self) -> bool {
                    #(#fields1)&*
                }
            }
        }
    }

    fn impl_one(self, algebra: &Algebra) -> Option<TokenStream> {
        if !self.contains(Blade::scalar()) {
            return None;
        }

        let output = self
            .iter_blades_unsorted(algebra)
            .flat_map(|lhs| {
                self.iter_blades_unsorted(algebra)
                    .map(move |rhs| algebra.geo(lhs, rhs))
            })
            .collect::<Option<Self>>();

        if output != Some(self) {
            return None;
        }

        let where_clause = quote! {
            where
                T: num_traits::Float,
        };

        let fields = self.iter_blades_unsorted(algebra).map(|b| {
            let field = b.field_ref(algebra);
            let value = if b == Blade::scalar() {
                quote! { num_traits::One::one() }
            } else {
                quote! { num_traits::Zero::zero() }
            };
            quote! { #field: #value }
        });

        Some(quote! {
            impl<T> num_traits::One for #self<T> #where_clause {
                fn one() -> Self {
                    #self {
                        #(#fields,)*
                    }
                }
            }
        })
    }

    pub fn fn_ident(self) -> Ident {
        Ident::new(self.name_lowercase(), Span::mixed_site())
    }

    fn has_impl_sample_unit(self, algebra: &Algebra) -> bool {
        matches!(self, Type::Grade(_))
            && !algebra.slim
            && !algebra.has_negative_bases()
            && self
                .iter_blades_unsorted(algebra)
                .any(|b| !algebra.dot(b, b).is_zero())
    }

    pub fn impl_sample_unit(self, algebra: &Algebra) -> Option<TokenStream> {
        if !self.has_impl_sample_unit(algebra) {
            return None;
        }

        let fields = self.iter_blades_unsorted(algebra).map(|b| {
            let f = b.field_ref(algebra);
            if algebra.dot(b, b).is_zero() {
                quote! {
                    #f: T::zero(),
                }
            } else {
                quote! {
                    #f: rng.gen::<T>() * two - T::one(),
                }
            }
        });

        let unit_ty = InverseOps::Unitize.trait_ty();
        let unit_fn = InverseOps::Unitize.trait_fn();
        let norm2_ty = NormOps::Norm2.trait_ty();
        let norm2_fn = NormOps::Norm2.trait_fn();

        Some(quote! {
            impl<T> rand::distributions::Distribution<#self<T>> for rand::distributions::Standard
            where
                T: num_traits::Float,
                rand::distributions::Standard: rand::distributions::Distribution<T>,
            {
                #[inline]
                fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> #self<T> {
                    let two = T::one() + T::one();
                    for _ in 0..64 {
                        let v = #self { #( #fields )* };
                        if #norm2_ty::#norm2_fn(v) <= T::one() {
                            return v;
                        }
                    }
                    panic!("unable to find unit value for {}", std::any::type_name::<Self>());
                }
            }
            impl<T> rand::distributions::Distribution<Unit<#self<T>>> for rand::distributions::Standard
            where
                T: num_traits::Float,
                rand::distributions::Standard: rand::distributions::Distribution<T>,
            {
                #[inline]
                fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Unit<#self<T>> {
                    #unit_ty::#unit_fn(rng.gen::<#self<T>>())
                }
            }
        })
    }

    fn test_sample_unit(self, algebra: &Algebra) -> Option<TokenStream> {
        if !self.has_impl_sample_unit(algebra) {
            return None;
        }

        let fn_ident = Ident::new(
            &format!("test_sample_unit_{}", self.name_lowercase()),
            Span::mixed_site(),
        );

        let norm2_ty = NormOps::Norm2.trait_ty();
        let norm2_fn = NormOps::Norm2.trait_fn();

        Some(quote! {
            #[test]
            fn #fn_ident() {
                let mut rng = rand::thread_rng();
                let v: Unit<#self<f64>> = rand::Rng::gen(&mut rng);
                assert!(f64::abs(1. - #norm2_ty::#norm2_fn(v.value()).s) < 1e-9);
            }
        })
    }
}

impl Complement {
    pub fn trait_ty(self) -> TokenStream {
        match self {
            Self::Dual => quote!(geo_traits::Dual),
            Self::LeftComp => quote!(geo_traits::LeftComplement),
            Self::RightComp => quote!(geo_traits::RightComplement),
        }
    }

    pub fn trait_fn(self) -> TokenStream {
        match self {
            Self::Dual => quote!(dual),
            Self::LeftComp => quote!(left_comp),
            Self::RightComp => quote!(right_comp),
        }
    }
}

impl ToTokens for Type {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ident().to_tokens(tokens);
    }
}

impl Blade {
    pub fn field_ref(self, algebra: &Algebra) -> &Ident {
        &algebra.fields[self.unsigned().0 as usize]
    }

    pub fn field(self, bases: &[Basis]) -> Ident {
        let mut output = String::new();
        let blade_bases = bases
            .iter()
            .enumerate()
            .filter_map(move |(i, b)| self.contains(i as u32).then_some(*b));

        for basis in blade_bases {
            output.push(basis.char);
        }
        if output.is_empty() {
            output.push('s');
        } else if output.chars().next().unwrap().is_numeric() {
            // idents cannot start with a number
            output.insert(0, 'e');
        }
        Ident::new(&output, Span::mixed_site())
    }
}

impl ProductOp {
    pub fn trait_ty(self) -> TokenStream {
        match self {
            ProductOp::Geo => quote!(geo_traits::Geo),
            ProductOp::Wedge => quote!(geo_traits::Wedge),
            ProductOp::Dot => quote!(geo_traits::Dot),
            ProductOp::Antigeo => quote!(geo_traits::Antigeo),
            ProductOp::Antidot => quote!(geo_traits::Antidot),
            ProductOp::Antiwedge => quote!(geo_traits::Antiwedge),
            ProductOp::Mul => quote!(std::ops::Mul),
            ProductOp::Commutator => quote!(geo_traits::Commutator),
        }
    }

    pub fn trait_fn(self) -> TokenStream {
        match self {
            ProductOp::Geo => quote!(geo),
            ProductOp::Wedge => quote!(wedge),
            ProductOp::Dot => quote!(dot),
            ProductOp::Antigeo => quote!(antigeo),
            ProductOp::Antidot => quote!(antidot),
            ProductOp::Antiwedge => quote!(antiwedge),
            ProductOp::Mul => quote!(mul),
            ProductOp::Commutator => quote!(com),
        }
    }

    pub fn has_implementation(self, lhs: Type, rhs: Type, algebra: &Algebra) -> bool {
        if !Self::iter_all(algebra).any(|op| op == self) {
            return false;
        }

        !algebra.slim
            || lhs.is_scalar()
            || rhs.is_scalar()
            || rhs.is_vector()
            || rhs == Type::pseudoscalar(algebra)
    }

    fn impl_for(self, algebra: &Algebra, lhs: Type, rhs: Type) -> Option<TokenStream> {
        let op = self.trait_ty();
        let op_fn = self.trait_fn();
        let output = self.output(algebra, lhs, rhs)?;

        let blades = cartesian_product(
            lhs.iter_blades_unsorted(algebra),
            rhs.iter_blades_unsorted(algebra),
        )
        .fold(
            HashMap::<Blade, Vec<(Blade, Blade, Blade)>>::new(),
            |mut map, (lhs, rhs)| {
                let output = self.product(algebra, lhs, rhs);
                if !output.is_zero() {
                    map.entry(output.unsigned())
                        .or_default()
                        .push((lhs, rhs, output));
                }
                map
            },
        );

        let fields = output.iter_blades_unsorted(algebra).map(|blade| {
            let ident = blade.field_ref(algebra);

            let mut sum = quote!();

            if let Some(vec) = blades.get(&blade) {
                for (l, r, o) in vec {
                    let lf = l.field_ref(algebra);
                    let rf = r.field_ref(algebra);
                    match (sum.is_empty(), o.is_positive()) {
                        (true, true) => sum.extend(quote! { self.#lf * rhs.#rf }),
                        (true, false) => sum.extend(quote! { -(self.#lf * rhs.#rf) }),
                        (false, true) => sum.extend(quote! { + self.#lf * rhs.#rf }),
                        (false, false) => sum.extend(quote! { - self.#lf * rhs.#rf }),
                    }
                }
            } else {
                sum.extend(quote! { num_traits::Zero::zero() });
            }

            quote! {
                #ident: #sum,
            }
        });

        let fn_attrs = fn_attrs();

        Some(quote! {
            impl<T, U, V> #op<#rhs<U>> for #lhs<T>
            where
                T: num_traits::Float + std::ops::Mul<U, Output = V>,
                U: num_traits::Float,
                V: num_traits::Float,
            {
                type Output = #output<V>;
                #fn_attrs
                fn #op_fn(self, rhs: #rhs<U>) -> Self::Output {
                    #output {
                        #(#fields)*
                    }
                }
            }
        })
    }
}

impl SumOp {
    fn impl_for(self, algebra: &Algebra, lhs: Type, rhs: Type) -> Option<TokenStream> {
        if algebra.slim && lhs != rhs {
            return None;
        }

        enum Side {
            Lhs,
            Rhs,
        }

        impl ToTokens for Side {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                match self {
                    Self::Lhs => tokens.extend(quote!(self)),
                    Self::Rhs => tokens.extend(quote!(rhs)),
                }
            }
        }

        let output = Self::sum(algebra, lhs, rhs)?;
        if !algebra.contains(output) {
            return None;
        }

        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();

        let mut blades = HashMap::<Blade, Vec<(Side, Blade)>>::new();

        for blade in lhs.iter_blades_unsorted(algebra) {
            blades
                .entry(blade.unsigned())
                .or_default()
                .push((Side::Lhs, blade));
        }
        for mut blade in rhs.iter_blades_unsorted(algebra) {
            if matches!(self, SumOp::Sub) {
                blade = -blade;
            }
            blades
                .entry(blade.unsigned())
                .or_default()
                .push((Side::Rhs, blade));
        }

        let fields = output.iter_blades_unsorted(algebra).map(|blade| {
            let field = blade.field_ref(algebra);
            let mut sum = quote!();
            if let Some(vec) = blades.get(&blade) {
                for (side, blade) in vec {
                    let f = blade.field_ref(algebra);
                    match (sum.is_empty(), blade.is_positive()) {
                        (true, true) => sum.extend(quote! { #side.#f }),
                        (true, false) => sum.extend(quote! { -#side.#f }),
                        (false, true) => sum.extend(quote! { + #side.#f }),
                        (false, false) => sum.extend(quote! { - #side.#f }),
                    }
                }
            } else {
                sum.extend(quote!(num_traits::Zero::zero()));
            }
            quote! { #field: #sum, }
        });

        let fn_attrs = fn_attrs();

        Some(quote! {
            impl<T> #trait_ty<#rhs<T>> for #lhs<T>
            where
                T: num_traits::Float,
            {
                type Output = #output<T>;
                #fn_attrs
                fn #trait_fn(self, rhs: #rhs<T>) -> Self::Output {
                    #output {
                        #(#fields)*
                    }
                }
            }
        })
    }

    pub fn trait_ty(self) -> TokenStream {
        match self {
            SumOp::Add => quote! { std::ops::Add },
            SumOp::Sub => quote! { std::ops::Sub },
        }
    }

    pub fn trait_fn(self) -> TokenStream {
        match self {
            SumOp::Add => quote! { add },
            SumOp::Sub => quote! { sub },
        }
    }
}

impl ScalarOps {
    pub fn impl_for_scalar(self, ty: Type, algebra: &Algebra) -> Vec<TokenStream> {
        if ty == Type::Mv {
            return vec![];
        }

        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();

        let fn_attrs = quote!(
            #[inline]
            #[allow(clippy::suspicious_arithmetic_impl)]
        );

        match self {
            Self::Div => {
                let inv_ty = InverseOps::Inverse.trait_ty();
                let inv_fn = InverseOps::Inverse.trait_fn();

                if InverseOps::Inverse.inapplicable(ty, algebra) {
                    vec![
                        quote! {
                            impl<T> #trait_ty<f32> for #ty<T>
                            where
                                T: std::ops::Mul<f32, Output = T>,
                            {
                                type Output = #ty<T>;
                                #fn_attrs
                                fn #trait_fn(self, rhs: f32) -> Self::Output {
                                    let recip = rhs.recip();
                                    self.map(|t| t * recip)
                                }
                            }
                        },
                        quote! {
                            impl<T> #trait_ty<f64> for #ty<T>
                            where
                                T: std::ops::Mul<f64, Output = T>,
                            {
                                type Output = #ty<T>;
                                #fn_attrs
                                fn #trait_fn(self, rhs: f64) -> Self::Output {
                                    let recip = rhs.recip();
                                    self.map(|t| t * recip)
                                }
                            }
                        },
                    ]
                } else {
                    vec![
                        quote! {
                            impl<T> #trait_ty<#ty<T>> for f32
                            where
                                T: num_traits::Float + std::ops::Mul<f32, Output = T>,
                            {
                                type Output = #ty<T>;
                                #fn_attrs
                                fn #trait_fn(self, rhs: #ty<T>) -> Self::Output {
                                    #inv_ty::#inv_fn(rhs).map(|t| t * self)
                                }
                            }
                        },
                        quote! {
                            impl<T> #trait_ty<#ty<T>> for f64
                            where
                                T: num_traits::Float + std::ops::Mul<f64, Output = T>,
                            {
                                type Output = #ty<T>;
                                #fn_attrs
                                fn #trait_fn(self, rhs: #ty<T>) -> Self::Output {
                                    #inv_ty::#inv_fn(rhs).map(|t| t * self)
                                }
                            }
                        },
                        quote! {
                            impl<T> #trait_ty<f32> for #ty<T>
                            where
                                T: std::ops::Mul<f32, Output = T>,
                            {
                                type Output = #ty<T>;
                                #fn_attrs
                                fn #trait_fn(self, rhs: f32) -> Self::Output {
                                    let recip = rhs.recip();
                                    self.map(|t| t * recip)
                                }
                            }
                        },
                        quote! {
                            impl<T> #trait_ty<f64> for #ty<T>
                            where
                                T: std::ops::Mul<f64, Output = T>,
                            {
                                type Output = #ty<T>;
                                #fn_attrs
                                fn #trait_fn(self, rhs: f64) -> Self::Output {
                                    let recip = rhs.recip();
                                    self.map(|t| t * recip)
                                }
                            }
                        },
                    ]
                }
            }
            Self::Mul => vec![
                quote! {
                    impl<T> #trait_ty<#ty<T>> for f32
                    where
                        T: #trait_ty<f32, Output = T>,
                    {
                        type Output = #ty<T>;
                        #fn_attrs
                        fn #trait_fn(self, rhs: #ty<T>) -> Self::Output {
                            #trait_ty::#trait_fn(rhs, self)
                        }
                    }
                },
                quote! {
                    impl<T> #trait_ty<#ty<T>> for f64
                    where
                        T: #trait_ty<f64, Output = T>,
                    {
                        type Output = #ty<T>;
                        #fn_attrs
                        fn #trait_fn(self, rhs: #ty<T>) -> Self::Output {
                            #trait_ty::#trait_fn(rhs, self)
                        }
                    }
                },
                quote! {
                    impl<T> #trait_ty<f32> for #ty<T>
                    where
                        T: #trait_ty<f32, Output = T>,
                    {
                        type Output = #ty<T>;
                        #fn_attrs
                        fn #trait_fn(self, rhs: f32) -> Self::Output {
                            self.map(|t| #trait_ty::#trait_fn(t, rhs))
                        }
                    }
                },
                quote! {
                    impl<T> #trait_ty<f64> for #ty<T>
                    where
                        T: #trait_ty<f64, Output = T>,
                    {
                        type Output = #ty<T>;
                        #fn_attrs
                        fn #trait_fn(self, rhs: f64) -> Self::Output {
                            self.map(|t| #trait_ty::#trait_fn(t, rhs))
                        }
                    }
                },
            ],
            Self::Add | Self::Sub => {
                let output = SumOp::sum(algebra, ty, Type::Grade(0));
                vec![
                    quote! {
                        impl #trait_ty<f32> for #ty<f32> {
                            type Output = #output<f32>;
                            #[inline]
                            fn #trait_fn(self, rhs: f32) -> Self::Output {
                                #trait_ty::#trait_fn(self, Scalar { s: rhs })
                            }
                        }
                    },
                    quote! {
                        impl #trait_ty<f64> for #ty<f64> {
                            type Output = #output<f64>;
                            #[inline]
                            fn #trait_fn(self, rhs: f64) -> Self::Output {
                                #trait_ty::#trait_fn(self, Scalar { s: rhs })
                            }
                        }
                    },
                    quote! {
                        impl #trait_ty<#ty<f32>> for f32 {
                            type Output = #output<f32>;
                            #[inline]
                            fn #trait_fn(self, rhs: #ty<f32>) -> Self::Output {
                                #trait_ty::#trait_fn(Scalar { s: self }, rhs)
                            }
                        }
                    },
                    quote! {
                        impl #trait_ty<#ty<f64>> for f64 {
                            type Output = #output<f64>;
                            #[inline]
                            fn #trait_fn(self, rhs: #ty<f64>) -> Self::Output {
                                #trait_ty::#trait_fn(Scalar { s: self }, rhs)
                            }
                        }
                    },
                ]
            }
        }
    }

    pub fn trait_ty(self) -> TokenStream {
        match self {
            Self::Mul => quote!(std::ops::Mul),
            Self::Div => quote!(std::ops::Div),
            Self::Add => quote!(std::ops::Add),
            Self::Sub => quote!(std::ops::Sub),
        }
    }

    pub fn trait_fn(self) -> TokenStream {
        match self {
            Self::Mul => quote!(mul),
            Self::Div => quote!(div),
            Self::Add => quote!(add),
            Self::Sub => quote!(sub),
        }
    }
}

impl ScalarAssignOps {
    pub fn impl_for(self, ty: Type, algebra: &Algebra) -> TokenStream {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let fn_attrs = fn_attrs();
        let fields = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field_ref(algebra);
            quote! { #trait_ty::#trait_fn(&mut self.#f, rhs); }
        });
        quote! {
            impl<T, U> #trait_ty<U> for #ty<T>
            where
                T: #trait_ty<U>,
                U: Copy,
            {
                #fn_attrs
                fn #trait_fn(&mut self, rhs: U) {
                    #(#fields)*
                }
            }
        }
    }

    pub fn trait_ty(self) -> TokenStream {
        match self {
            Self::Mul => quote!(std::ops::MulAssign),
            Self::Div => quote!(std::ops::DivAssign),
        }
    }

    pub fn trait_fn(self) -> TokenStream {
        match self {
            Self::Mul => quote!(mul_assign),
            Self::Div => quote!(div_assign),
        }
    }
}

impl SumAssignOps {
    pub fn impl_for(self, ty: Type, algebra: &Algebra) -> TokenStream {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let fn_attrs = fn_attrs();
        let fields = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field_ref(algebra);
            quote! { #trait_ty::#trait_fn(&mut self.#f, rhs.#f); }
        });
        quote! {
            impl<T, U> #trait_ty<#ty<U>> for #ty<T>
            where
                T: #trait_ty<U>,
            {
                #fn_attrs
                fn #trait_fn(&mut self, rhs: #ty<U>) {
                    #(#fields)*
                }
            }
        }
    }

    pub fn trait_ty(self) -> TokenStream {
        match self {
            Self::AddAssign => quote!(std::ops::AddAssign),
            Self::SubAssign => quote!(std::ops::SubAssign),
        }
    }

    pub fn trait_fn(self) -> TokenStream {
        match self {
            Self::AddAssign => quote!(add_assign),
            Self::SubAssign => quote!(sub_assign),
        }
    }
}

impl FloatConversion {
    pub fn impl_for(self, ty: Type, algebra: &Algebra) -> TokenStream {
        let from = self.type_from();
        let to = self.type_to();
        let fn_ident = self.fn_ident();
        let fields1 = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field_ref(algebra);
            quote! {
                #f: self.#f as #to
            }
        });
        let fn_attrs = fn_attrs();
        quote! {
            impl #ty<#from> {
                #fn_attrs
                pub const fn #fn_ident(self) -> #ty<#to> {
                    #ty {
                        #(#fields1),*
                    }
                }
            }
        }
    }

    fn type_from(self) -> TokenStream {
        match self {
            Self::ToF64 => quote!(f32),
            Self::ToF32 => quote!(f64),
        }
    }

    fn type_to(self) -> TokenStream {
        match self {
            Self::ToF32 => quote!(f32),
            Self::ToF64 => quote!(f64),
        }
    }

    fn fn_ident(self) -> TokenStream {
        match self {
            Self::ToF32 => quote!(to_f32),
            Self::ToF64 => quote!(to_f64),
        }
    }
}

impl NormOps {
    pub fn impl_for(self, ty: Type, algebra: &Algebra) -> Option<TokenStream> {
        // Float::sqrt needed for Norm
        if self == NormOps::Norm && InverseOps::Inverse.inapplicable(ty, algebra) {
            return None;
        }

        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let fn_attrs = fn_attrs();

        let grade_product = GradeProduct::trait_ty();
        let rev_ty = Reverse::trait_ty();
        let rev_fn = Reverse::trait_fn();

        if !ty.iter_blades_unsorted(algebra).any(|blade| {
            let out = algebra.geo(blade, blade);
            out.grade() == 0 && !out.is_zero()
        }) {
            return None;
        }

        match self {
            NormOps::Norm2 => Some(quote! {
                impl<T> #trait_ty for #ty<T>
                where
                    T: num_traits::Float,
                {
                    type Output = Scalar<T>;
                    #fn_attrs
                    fn #trait_fn(self) -> Self::Output {
                        <Scalar<T> as #grade_product<_, _>>::product(self, #rev_ty::#rev_fn(self))
                    }
                }
            }),
            NormOps::Norm => Some(quote! {
                impl<T> #trait_ty for #ty<T>
                where
                    T: num_traits::Float,
                {
                    type Output = Scalar<T>;
                    #fn_attrs
                    fn #trait_fn(self) -> Self::Output {
                        let scalar = <Scalar<T> as #grade_product<_, _>>::product(self, #rev_ty::#rev_fn(self));
                        num_traits::Float::sqrt(scalar)
                    }
                }
            }),
        }
    }

    pub fn trait_ty(self) -> TokenStream {
        match self {
            NormOps::Norm => quote!(geo_traits::Norm),
            NormOps::Norm2 => quote!(geo_traits::Norm2),
        }
    }

    pub fn trait_fn(self) -> TokenStream {
        match self {
            NormOps::Norm => quote!(norm),
            NormOps::Norm2 => quote!(norm2),
        }
    }
}

pub struct Sandwich;

impl Sandwich {
    pub fn has_implementation(lhs: Type, rhs: Type, algebra: &Algebra) -> bool {
        let excluded = [Type::Grade(0), Type::pseudoscalar(algebra), Type::Mv];

        let Some(intermediate) = ProductOp::Geo.output(algebra, lhs, rhs) else { return false; };

        !excluded.contains(&lhs)
            && !excluded.contains(&rhs)
            && !InverseOps::Inverse.inapplicable(lhs, algebra)
            && !algebra.slim
            && GradeProduct::has_implementation(intermediate, lhs, rhs, algebra)
    }

    pub fn impl_for(lhs: Type, rhs: Type, algebra: &Algebra) -> Option<TokenStream> {
        if !Sandwich::has_implementation(lhs, rhs, algebra) {
            return None;
        }

        let intermediate = ProductOp::Geo.output(algebra, lhs, rhs)?;

        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();
        let fn_attrs = fn_attrs();
        let geo_ty = ProductOp::Geo.trait_ty();
        let geo_fn = ProductOp::Geo.trait_fn();
        let inv_ty = InverseOps::Inverse.trait_ty();
        let inv_fn = InverseOps::Inverse.trait_fn();
        let grade_prod_ty = GradeProduct::trait_ty();
        let grade_prod_fn = GradeProduct::trait_fn();

        let final_expr = match rhs {
            Type::Motor | Type::Flector => quote!(intermediate * #inv_ty::#inv_fn(self)),
            _ => quote! {
                <#rhs<V> as #grade_prod_ty<_, _>>::#grade_prod_fn(intermediate, #inv_ty::#inv_fn(self))
            },
        };

        Some(quote! {
            impl<T, U, V> #trait_ty<#rhs<U>> for #lhs<T>
            where
                T: num_traits::Float + std::ops::Mul<U, Output = V>,
                U: num_traits::Float,
                V: num_traits::Float + std::ops::Mul<T, Output = V>,
            {
                type Output = #rhs<V>;
                #fn_attrs
                fn #trait_fn(self, rhs: #rhs<U>) -> Self::Output {
                    let intermediate: #intermediate<V> = #geo_ty::#geo_fn(self, rhs);
                    #final_expr
                }
            }
        })
    }

    pub fn trait_ty() -> TokenStream {
        quote!(geo_traits::Sandwich)
    }

    pub fn trait_fn() -> TokenStream {
        quote!(sandwich)
    }
}

struct Antisandwich;

impl Antisandwich {
    pub fn impl_for(lhs: Type, rhs: Type, algebra: &Algebra) -> Option<TokenStream> {
        if !Sandwich::has_implementation(lhs, rhs, algebra) {
            return None;
        }

        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();
        let fn_attrs = fn_attrs();

        let sandwich_ty = Sandwich::trait_ty();
        let sandwich_fn = Sandwich::trait_fn();

        let dual_ty = Complement::Dual.trait_ty();
        let dual_fn = Complement::Dual.trait_fn();

        let l_comp_ty = Complement::LeftComp.trait_ty();
        let l_comp_fn = Complement::LeftComp.trait_fn();

        let r_comp_ty = Complement::RightComp.trait_ty();
        let r_comp_fn = Complement::RightComp.trait_fn();

        if algebra.symmetrical_complements() {
            Some(quote! {
                impl<T, U, V> #trait_ty<#rhs<U>> for #lhs<T>
                where
                    T: num_traits::Float + std::ops::Mul<U, Output = V>,
                    U: num_traits::Float,
                    V: num_traits::Float + std::ops::Mul<T, Output = V>,
                {
                    type Output = #rhs<V>;
                    #fn_attrs
                    fn #trait_fn(self, rhs: #rhs<U>) -> Self::Output {
                        let lhs = #dual_ty::#dual_fn(self);
                        let rhs = #dual_ty::#dual_fn(rhs);
                        let sandwich = #sandwich_ty::#sandwich_fn(lhs, rhs);
                        #dual_ty::#dual_fn(sandwich)
                    }
                }
            })
        } else {
            Some(quote! {
                impl<T, U, V> #trait_ty<#rhs<U>> for #lhs<T>
                where
                    T: num_traits::Float + std::ops::Mul<U, Output = V>,
                    U: num_traits::Float,
                    V: num_traits::Float + std::ops::Mul<T, Output = V>,
                {
                    type Output = #rhs<V>;
                    #fn_attrs
                    fn #trait_fn(self, rhs: #rhs<U>) -> Self::Output {
                        let lhs = #l_comp_ty::#l_comp_fn(self);
                        let rhs = #l_comp_ty::#l_comp_fn(rhs);
                        let sandwich = #sandwich_ty::#sandwich_fn(lhs, rhs);
                        #r_comp_ty::#r_comp_fn(sandwich)
                    }
                }
            })
        }
    }

    pub fn trait_ty() -> TokenStream {
        quote!(geo_traits::Antisandwich)
    }

    pub fn trait_fn() -> TokenStream {
        quote!(antisandwich)
    }
}

impl InverseOps {
    pub fn trait_ty(self) -> TokenStream {
        match self {
            InverseOps::Inverse => quote!(geo_traits::Inverse),
            InverseOps::Unitize => quote!(geo_traits::Unitize),
        }
    }

    pub fn trait_fn(self) -> TokenStream {
        match self {
            InverseOps::Inverse => quote!(inv),
            InverseOps::Unitize => quote!(unit),
        }
    }

    /// No implementation if there's only one blade, or mv which are not easily inverted
    pub fn inapplicable(self, ty: Type, algebra: &Algebra) -> bool {
        ty == Type::Mv || ProductOp::Dot.output(algebra, ty, ty).is_none()
    }

    pub fn impl_for(self, ty: Type, algebra: &Algebra) -> Option<TokenStream> {
        if self.inapplicable(ty, algebra) {
            return None;
        }

        let scalar_fields = ty
            .iter_blades_unsorted(algebra)
            .filter(|b| {
                let dot = algebra.dot(*b, *b);
                !dot.is_zero() && dot.grade() == 0
            })
            .collect::<Vec<_>>();

        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();

        let fn_attrs = fn_attrs();

        let inv_ty = InverseOps::Inverse.trait_ty();
        let inv_fn = InverseOps::Inverse.trait_fn();

        let rev_ty = Reverse::trait_ty();
        let rev_fn = Reverse::trait_fn();

        let norm2_ty = NormOps::Norm2.trait_ty();
        let norm2_fn = NormOps::Norm2.trait_fn();

        let norm_ty = NormOps::Norm.trait_ty();
        let norm_fn = NormOps::Norm.trait_fn();

        Some(match (self, scalar_fields.len()) {
            (InverseOps::Inverse, 1) => {
                let field = scalar_fields[0].field_ref(algebra);
                quote! {
                    impl<T> #trait_ty for #ty<T>
                    where
                        T: num_traits::Float,
                    {
                        type Output = #ty<T>;
                        #fn_attrs
                        fn #trait_fn(self) -> Self::Output {
                            let norm2 = self.#field * self.#field;
                            let recip = num_traits::Float::recip(norm2);
                            #rev_ty::#rev_fn(self).map(|t| t * recip)
                        }
                    }
                }
            }
            (InverseOps::Inverse, _) => {
                quote! {
                    impl<T> #trait_ty for #ty<T>
                    where
                        T: num_traits::Float,
                    {
                        type Output = #ty<T>;
                        #fn_attrs
                        fn #trait_fn(self) -> Self::Output {
                            #rev_ty::#rev_fn(self) * #inv_ty::#inv_fn(#norm2_ty::#norm2_fn(self))
                        }
                    }
                }
            }
            (InverseOps::Unitize, 1) => {
                let field = scalar_fields[0].field_ref(algebra);
                quote! {
                    impl<T> #trait_ty for #ty<T>
                    where
                        T: num_traits::Float,
                    {
                        type Output = Unit<#ty<T>>;
                        #fn_attrs
                        fn #trait_fn(self) -> Self::Output {
                            let recip = self.#field.recip();
                            let value = self.map(|t| t * recip);
                            Unit(value)
                        }
                    }
                }
            }
            (InverseOps::Unitize, _) => {
                quote! {
                    impl<T> #trait_ty for #ty<T>
                    where
                        T: num_traits::Float,
                    {
                        type Output = Unit<#ty<T>>;
                        #fn_attrs
                        fn #trait_fn(self) -> Self::Output {
                            let norm = #norm_ty::#norm_fn(self);
                            let recip = num_traits::Float::recip(norm);
                            Unit(self * recip)
                        }
                    }
                }
            }
        })
    }

    pub fn tests(self, ty: Type, algebra: &Algebra) -> Option<TokenStream> {
        if self.inapplicable(ty, algebra) || algebra.slim || algebra.has_negative_bases() {
            return None;
        }

        let fn_ident = {
            let fn_name = match self {
                InverseOps::Inverse => format!("{}_inv_test", ty.name_lowercase()),
                InverseOps::Unitize => format!("{}_unit_test", ty.name_lowercase()),
            };
            Ident::new(&fn_name, Span::mixed_site())
        };

        let fields = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field_ref(algebra);
            quote! { #f: rng.gen::<f64>(), }
        });

        let geo_ty = ProductOp::Geo.trait_ty();
        let geo_fn = ProductOp::Geo.trait_fn();

        let inv_ty = InverseOps::Inverse.trait_ty();
        let inv_fn = InverseOps::Inverse.trait_fn();

        let unit_ty = InverseOps::Unitize.trait_ty();
        let unit_fn = InverseOps::Unitize.trait_fn();

        let norm2_ty = NormOps::Norm2.trait_ty();
        let norm2_fn = NormOps::Norm2.trait_fn();

        let rev_ty = Reverse::trait_ty();
        let rev_fn = Reverse::trait_fn();

        match self {
            InverseOps::Inverse => Some(quote! {
                #[test]
                fn #fn_ident() {
                    use rand::Rng;
                    let mut success_count = 0;
                    const N: usize = 1000;
                    let mut rng = rand::thread_rng();
                    for _ in 0..N {
                        let value = #ty {
                            #(#fields)*
                        };
                        let inv = #inv_ty::#inv_fn(value);
                        let product = #geo_ty::#geo_fn(value, inv);
                        let remainder = #norm2_ty::#norm2_fn(product - Scalar { s: 1. });
                        if remainder.s < 1e-10 {
                            success_count += 1;
                        }
                    }
                    assert_eq!(N, success_count);
                }
            }),
            InverseOps::Unitize => Some(quote! {
                #[test]
                fn #fn_ident() {
                    use rand::Rng;
                    let mut success_count = 0;
                    const N: usize = 1000;
                    let mut rng = rand::thread_rng();
                    for _ in 0..N {
                        let value = #ty {
                            #(#fields)*
                        };
                        let unit = #unit_ty::#unit_fn(value).value();
                        let product = #geo_ty::#geo_fn(unit, #rev_ty::#rev_fn(unit));

                        let remainder = #norm2_ty::#norm2_fn(product - Scalar { s: 1. });
                        if remainder.s.abs() < 1e-10 {
                            success_count += 1;
                            continue;
                        }

                        let remainder = #norm2_ty::#norm2_fn(product + Scalar { s: 1. });
                        if remainder.s.abs() < 1e-10 {
                            success_count += 1;
                            continue;
                        }
                    }
                    assert_eq!(N, success_count);
                }
            }),
        }
    }
}

pub struct Unit;

impl Unit {
    pub fn define(algebra: &Algebra) -> Vec<TokenStream> {
        let unitize_ty = InverseOps::Unitize.trait_ty();
        let unitize_fn = InverseOps::Unitize.trait_fn();
        let inverse_ty = InverseOps::Inverse.trait_ty();
        let inverse_fn = InverseOps::Inverse.trait_fn();
        let reverse_ty = Reverse::trait_ty();
        let item_struct = quote! {
            #[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
            pub struct Unit<T>(T);
        };
        let item_impl = quote! {
            impl<T> Unit<T> {
                #[inline]
                pub const fn assert(value: T) -> Unit<T> {
                    Unit(value)
                }
                #[inline]
                pub fn value(self) -> T {
                    self.0
                }
            }
        };
        let unitize_impl = quote! {
            impl<T> #unitize_ty for Unit<T> where T: #unitize_ty {
                type Output = Self;
                #[inline]
                fn #unitize_fn(self) -> Self::Output {
                    self
                }
            }
        };
        let inverse_impl = quote! {
            impl<T> #inverse_ty for Unit<T> where T: #reverse_ty<Output = T> {
                type Output = Self;
                #[inline]
                fn #inverse_fn(self) -> Self::Output {
                    Unit(self.0.rev())
                }
            }
        };
        let geo_ty = ProductOp::Geo.trait_ty();
        let geo_fn = ProductOp::Geo.trait_fn();
        let sandwich_ty = Sandwich::trait_ty();
        let sandwich_fn = Sandwich::trait_fn();
        let rev_ty = Reverse::trait_ty();
        let grade_prod_ty = GradeProduct::trait_ty();

        let sandwich_impl_lhs = quote! {
            impl<Lhs, Rhs, Int> #sandwich_ty<Rhs> for Unit<Lhs>
            where
                Lhs: #geo_ty<Rhs, Output = Int> + #rev_ty<Output = Lhs> + Copy,
                Rhs: #grade_prod_ty<Int, Lhs, Output = Rhs>,
            {
                type Output = Rhs;
                #[inline]
                fn #sandwich_fn(self, rhs: Rhs) -> Self::Output {
                    let int = #geo_ty::#geo_fn(self.value(), rhs);
                    Rhs::product(int, self.value().rev())
                }
            }
        };

        let sandwich_impl_rhs = algebra.type_tuples().map(|(lhs, rhs)| {
            quote! {
                impl<T, U, V> #sandwich_ty<Unit<#rhs<U>>> for #lhs<T>
                where
                    #lhs<T>: #sandwich_ty<#rhs<U>, Output = #rhs<V>>,
                {
                    type Output = Unit<#rhs<V>>;
                    #[inline]
                    fn #sandwich_fn(self, rhs: Unit<#rhs<U>>) -> Self::Output {
                        let output = #sandwich_ty::#sandwich_fn(self, rhs.value());
                        Unit::assert(output)
                    }
                }
            }
        });

        let sandwich_impl_both = algebra.type_tuples().map(|(lhs, rhs)| {
            quote! {
                impl<T, U, V> #sandwich_ty<Unit<#rhs<U>>> for Unit<#lhs<T>>
                where
                    Unit<#lhs<T>>: #sandwich_ty<#rhs<U>, Output = #rhs<V>>,
                {
                    type Output = Unit<#rhs<V>>;
                    #[inline]
                    fn #sandwich_fn(self, rhs: Unit<#rhs<U>>) -> Self::Output {
                        let output = #sandwich_ty::#sandwich_fn(self, rhs.value());
                        Unit::assert(output)
                    }
                }
            }
        });

        let operator_overloads = algebra
            .types()
            .flat_map(|ty| Overload::iter(algebra).filter_map(move |op| op.impl_for_unit(ty)));

        IntoIterator::into_iter([
            item_struct,
            item_impl,
            unitize_impl,
            inverse_impl,
            sandwich_impl_lhs,
        ])
        .chain(sandwich_impl_rhs)
        .chain(sandwich_impl_both)
        .chain(ProductOp::iter_all(algebra).map(|op| {
            let trait_ty = op.trait_ty();
            let trait_fn = op.trait_fn();
            quote! {
                impl<Lhs, Rhs, Output> #trait_ty<Rhs> for Unit<Lhs>
                where
                    Lhs: #trait_ty<Rhs, Output = Output>
                {
                    type Output = Output;
                    #[inline]
                    fn #trait_fn(self, rhs: Rhs) -> Self::Output {
                        #trait_ty::#trait_fn(self.value(), rhs)
                    }
                }
            }
        }))
        .chain(operator_overloads)
        .collect()
    }
}

pub struct Div;

impl Div {
    pub fn impl_for(lhs: Type, rhs: Type, algebra: &Algebra) -> Option<TokenStream> {
        if InverseOps::Inverse.inapplicable(rhs, algebra) {
            return None;
        }

        let output = lhs
            .iter_blades_unsorted(algebra)
            .flat_map(|lhs| {
                rhs.iter_blades_unsorted(algebra)
                    .map(move |rhs| algebra.geo(lhs, rhs))
            })
            .collect::<Option<Type>>()?;

        if !algebra.contains(output) {
            return None;
        }

        let inv_ty = InverseOps::Inverse.trait_ty();
        let inv_fn = InverseOps::Inverse.trait_fn();

        Some(quote! {
            impl<T, U, V> std::ops::Div<#rhs<U>> for #lhs<T>
            where
                T: num_traits::Float + std::ops::Mul<U, Output = V>,
                U: num_traits::Float,
                V: num_traits::Float,
            {
                type Output = #output<V>;
                #[inline]
                #[allow(clippy::suspicious_arithmetic_impl)]
                fn div(self, rhs: #rhs<U>) -> Self::Output {
                    self * #inv_ty::#inv_fn(rhs)
                }
            }
        })
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum UnaryOverload {
    Not,
}

impl UnaryOverload {
    pub fn iter(algebra: &Algebra) -> impl Iterator<Item = Self> {
        if algebra.symmetrical_complements() {
            Some(Self::Not).into_iter()
        } else {
            None.into_iter()
        }
    }

    pub fn impl_for(self, ty: Type, algebra: &Algebra) -> TokenStream {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let output = ty.complement(algebra);
        let inner_ty = self.inner_ty();
        let inner_fn = self.inner_fn();
        quote! {
            impl<T> #trait_ty for #ty<T> where T: num_traits::Float {
                type Output = #output<T>;
                #[inline]
                fn #trait_fn(self) -> Self::Output {
                    #inner_ty::#inner_fn(self)
                }
            }
        }
    }

    pub fn trait_ty(self) -> TokenStream {
        match self {
            Self::Not => quote!(std::ops::Not),
        }
    }

    pub fn trait_fn(self) -> TokenStream {
        match self {
            Self::Not => quote!(not),
        }
    }

    pub fn inner_ty(self) -> TokenStream {
        match self {
            Self::Not => quote!(geo_traits::Dual),
        }
    }

    pub fn inner_fn(self) -> TokenStream {
        match self {
            Self::Not => quote!(dual),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Overload {
    And,
    Or,
    Xor,
    Shr,
}

impl Overload {
    pub fn iter(algebra: &Algebra) -> impl Iterator<Item = Self> {
        if algebra.slim {
            IntoIterator::into_iter(vec![Overload::Or, Overload::Xor])
        } else {
            IntoIterator::into_iter(vec![
                Overload::And,
                Overload::Or,
                Overload::Xor,
                Overload::Shr,
            ])
        }
    }

    pub fn impl_for(self, lhs: Type, rhs: Type, algebra: &Algebra) -> Option<TokenStream> {
        match self.inner_op() {
            OverloadOp::Sandwich => {
                if !Sandwich::has_implementation(lhs, rhs, algebra) {
                    return None;
                }
            }
            OverloadOp::Product(op) => {
                if !op.has_implementation(lhs, rhs, algebra) {
                    return None;
                }
            }
        }

        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let inner = self.inner_op();
        let inner_ty = inner.trait_ty();
        let inner_fn = inner.trait_fn();
        let output = inner.output(algebra, lhs, rhs)?;
        if !algebra.contains(output) {
            return None;
        }

        let where_clause = match self {
            Overload::Shr => {
                quote! {
                    where
                        T: num_traits::Float + std::ops::Mul<U, Output = V>,
                        U: num_traits::Float,
                        V: num_traits::Float + std::ops::Mul<T, Output = V>,
                }
            }
            _ => quote! {
                where
                    T: num_traits::Float + std::ops::Mul<U, Output = V>,
                    U: num_traits::Float,
                    V: num_traits::Float,
            },
        };

        Some(quote! {
            impl<T, U, V> #trait_ty<#rhs<U>> for #lhs<T> #where_clause {
                type Output = #output<V>;
                #[inline]
                fn #trait_fn(self, rhs: #rhs<U>) -> Self::Output {
                    #inner_ty::#inner_fn(self, rhs)
                }
            }
        })
    }

    pub fn impl_for_unit(self, lhs: Type) -> Option<TokenStream> {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let inner = self.inner_op();
        let inner_ty = inner.trait_ty();
        let inner_fn = inner.trait_fn();
        Some(quote! {
            impl<T, U, V> #trait_ty<U> for Unit<#lhs<T>>
            where
                Unit<#lhs<T>>: #inner_ty<U, Output = V>,
            {
                type Output = V;
                #[inline]
                fn #trait_fn(self, rhs: U) -> Self::Output {
                    #inner_ty::#inner_fn(self, rhs)
                }
            }
        })
    }

    pub fn trait_ty(self) -> TokenStream {
        match self {
            Overload::And => quote!(std::ops::BitAnd),
            Overload::Or => quote!(std::ops::BitOr),
            Overload::Xor => quote!(std::ops::BitXor),
            Overload::Shr => quote!(std::ops::Shr),
        }
    }

    pub fn trait_fn(self) -> TokenStream {
        match self {
            Overload::And => quote!(bitand),
            Overload::Or => quote!(bitor),
            Overload::Xor => quote!(bitxor),
            Overload::Shr => quote!(shr),
        }
    }

    pub fn inner_op(self) -> OverloadOp {
        match self {
            Self::And => OverloadOp::Product(ProductOp::Antiwedge),
            Self::Or => OverloadOp::Product(ProductOp::Dot),
            Self::Xor => OverloadOp::Product(ProductOp::Wedge),
            Self::Shr => OverloadOp::Sandwich,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum OverloadOp {
    Product(ProductOp),
    Sandwich,
}

impl OverloadOp {
    fn trait_ty(self) -> TokenStream {
        match self {
            Self::Product(op) => op.trait_ty(),
            Self::Sandwich => Sandwich::trait_ty(),
        }
    }
    fn trait_fn(self) -> TokenStream {
        match self {
            Self::Product(op) => op.trait_fn(),
            Self::Sandwich => Sandwich::trait_fn(),
        }
    }
    fn output(self, algebra: &Algebra, lhs: Type, rhs: Type) -> Option<Type> {
        match self {
            OverloadOp::Product(op) => op.output(algebra, lhs, rhs),
            OverloadOp::Sandwich => Some(rhs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::ToTokens;

    impl Algebra {
        pub fn test3() -> Self {
            Algebra::new(vec![
                Basis {
                    char: 'x',
                    sqr: Square::Pos,
                },
                Basis {
                    char: 'y',
                    sqr: Square::Zero,
                },
                Basis {
                    char: 'z',
                    sqr: Square::Neg,
                },
            ])
        }
    }

    #[test]
    fn define_type() {
        let tokens = Type::Grade(2)
            .define(&crate::algebra::tests::ga_3d())
            .to_token_stream()
            .to_string();
        let expected = quote! {
            #[repr(C)]
            #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
            pub struct Bivector<T> {
                pub xy: T,
                pub xz: T,
                pub yz: T,
            }
        }
        .to_string();
        assert_eq!(expected, tokens);
    }

    #[test]
    fn numeric_bases() {
        let a = Algebra::new(vec![Basis {
            char: '1',
            sqr: Square::Pos,
        }]);

        assert_eq!(
            "e1",
            Blades::from(&a).nth(1).unwrap().field_ref(&a).to_string()
        );
    }
}

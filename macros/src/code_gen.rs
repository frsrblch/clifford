use crate::algebra::*;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use std::collections::HashMap;
use syn::{parse_quote, ItemFn, ItemImpl, ItemStruct};

impl Algebra {
    pub fn define(self) -> TokenStream {
        let structs = self.types().map(move |ty| ty.define(self));

        let scalar_float = impl_scalar_float();

        let impl_new_fn = self.types().map(|ty| ty.impl_new_fn(self));
        let impl_map_fn = self.types().map(|ty| ty.impl_map(self));
        let impl_grade_fns = self.types().filter_map(|ty| ty.impl_grade_fns(self));
        let impl_from = self.types().flat_map(|ty| ty.impl_from(self));

        let impl_zero = self.types().map(|ty| ty.impl_zero(self));
        let impl_one = self.types().filter_map(|ty| ty.impl_one(self));
        let impl_bytemuck = self.types().map(ImplBytemuck);

        let impl_product_ops = self.type_tuples().flat_map(|(lhs, rhs)| {
            ProductOp::iter_all(self).filter_map(move |op| op.impl_for(self, lhs, rhs))
        });

        let operator_overloads = self.type_tuples().flat_map(|(lhs, rhs)| {
            Overload::iter(self).filter_map(move |op| op.impl_for(lhs, rhs, self))
        });

        let div_ops = self
            .type_tuples()
            .filter_map(|(lhs, rhs)| Div::impl_for(lhs, rhs, self));

        let impl_sum_ops = SumOp::iter().flat_map(|op| {
            self.type_tuples()
                .filter_map(move |(lhs, rhs)| op.impl_for(self, lhs, rhs))
        });

        let impl_neg = self.types().map(|ty| Neg::impl_for(ty, self));

        let impl_rev = self.types().map(|ty| Reverse::impl_for(ty, self));
        let impl_antirev = self.types().map(|ty| Antireverse::impl_for(ty, self));

        let impl_complements = Complement::iter(self)
            .flat_map(|comp| self.types().map(move |ty| ty.impl_complement(self, comp)));

        let explicit_scalar_ops = ScalarOps::iter()
            .flat_map(|op| {
                self.types()
                    .filter_map(move |ty| op.impl_for_scalar(ty, self))
            })
            .flatten();

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

        let grade_antiproducts = self.grades().flat_map(|out| {
            self.type_tuples()
                .map(move |(lhs, rhs)| GradeAntiproduct::impl_for(lhs, rhs, out, self))
        });

        let norm_ops =
            NormOps::iter().flat_map(|op| self.types().map(move |ty| op.impl_for(ty, self)));

        let sandwich_ops = self
            .type_tuples()
            .map(|(lhs, rhs)| Sandwich::impl_for(lhs, rhs, self));

        let antisandwich_ops = self
            .type_tuples()
            .map(|(lhs, rhs)| Antisandwich::impl_for(lhs, rhs, self));

        let trig_ops = TrigOps::iter().map(|op| op.impl_for_scalar());

        let unit_items = Unit::define(self);

        let sample_unit = self.types().filter_map(|ty| ty.impl_sample_unit(self));

        let inverse_ops = InverseOps::iter()
            .flat_map(|op| self.types().filter_map(move |ty| op.impl_for(ty, self)));

        let impl_sqrt = self.types().map(move |ty| Sqrt::impl_for(ty, self));

        let test_unit_inv =
            InverseOps::iter().flat_map(|op| self.types().filter_map(move |ty| op.tests(ty, self)));

        let test_sample_unit = self.types().map(|ty| ty.test_sample_unit(self));

        quote!(
            #(#structs)*
            #scalar_float
            #(#impl_new_fn)*
            #(#impl_map_fn)*
            #(#impl_grade_fns)*
            #(#impl_from)*
            #(#impl_zero)*
            #(#impl_one)*
            #(#impl_bytemuck)*
            #(#impl_product_ops)*
            #(#operator_overloads)*
            #(#div_ops)*
            #(#impl_sum_ops)*
            #(#impl_neg)*
            #(#impl_rev)*
            #(#impl_antirev)*
            #(#impl_sqrt)*
            #(#impl_complements)*
            #(#explicit_scalar_ops)*
            #(#scalar_assign_ops)*
            #(#sum_assign_ops)*
            #(#float_conversion)*
            #(#grade_products)*
            #(#grade_antiproducts)*
            #(#norm_ops)*
            #(#sandwich_ops)*
            #(#antisandwich_ops)*
            #(#inverse_ops)*
            #(#trig_ops)*
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

fn impl_scalar_float() -> TokenStream {
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
    
        impl<T> num_traits::NumCast for Scalar<T>
        where
            T: num_traits::NumCast + num_traits::ToPrimitive,
        {
            fn from<N: num_traits::ToPrimitive>(n: N) -> Option<Self> {
                T::from(n).map(|s| Scalar { s })
            }
        }
    
        impl<T> num_traits::Num for Scalar<T>
        where
            T: num_traits::Num + std::ops::Neg<Output = T> + Copy,
        {
            type FromStrRadixErr = T::FromStrRadixErr;
            #[inline]
            fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                T::from_str_radix(str, radix).map(|s| Scalar { s })
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
        #[allow(unused_variables)]
    )
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TrigOps {
    Sin,
    Cos,
    Tan,
    Arcsin,
    Arccos,
    Arctan,
}

impl TrigOps {
    pub fn iter() -> impl Iterator<Item = Self> {
        IntoIterator::into_iter([
            Self::Sin,
            Self::Cos,
            Self::Tan,
            Self::Arcsin,
            Self::Arccos,
            Self::Arctan,
        ])
    }

    pub fn impl_for_scalar(self) -> ItemImpl {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let scalar = Type::Grade(0);
        let fn_attrs = fn_attrs();
        parse_quote! {
            impl<T, U> #trait_ty for #scalar<T>
            where
                T: #trait_ty<Output = U>
            {
                type Output = #scalar<U>;
                #fn_attrs
                fn #trait_fn(self) -> Self::Output {
                    self.map(#trait_ty::#trait_fn)
                }
            }
        }
    }

    pub fn trait_ty(self) -> syn::Type {
        match self {
            Self::Sin => parse_quote!(num_trig::Sin),
            Self::Cos => parse_quote!(num_trig::Cos),
            Self::Tan => parse_quote!(num_trig::Tan),
            Self::Arcsin => parse_quote!(num_trig::Arcsin),
            Self::Arccos => parse_quote!(num_trig::Arccos),
            Self::Arctan => parse_quote!(num_trig::Arctan),
        }
    }

    pub fn trait_fn(self) -> Ident {
        match self {
            Self::Sin => parse_quote!(sin),
            Self::Cos => parse_quote!(cos),
            Self::Tan => parse_quote!(tan),
            Self::Arcsin => parse_quote!(asin),
            Self::Arccos => parse_quote!(acos),
            Self::Arctan => parse_quote!(atan),
        }
    }
}

pub struct Neg;

impl Neg {
    fn impl_for(ty: Type, algebra: Algebra) -> ItemImpl {
        let fn_attrs = fn_attrs();
        let fields = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            quote! { #f: -self.#f, }
        });
        parse_quote! {
            impl<T> std::ops::Neg for #ty<T>
            where
                T: std::ops::Neg<Output = T>,
            {
                type Output = #ty<T>;
                #fn_attrs
                fn neg(self) -> Self::Output {
                    #ty {
                        #(#fields)*
                    }
                }
            }
        }
    }
}

pub struct Reverse;

impl Reverse {
    fn impl_for(ty: Type, algebra: Algebra) -> ItemImpl {
        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();
        let fn_attrs = fn_attrs();

        let fields = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            let rev = blade.rev();
            if rev.is_positive() {
                quote! { #f: self.#f, }
            } else {
                quote! { #f: -self.#f, }
            }
        });

        parse_quote! {
            impl<T> #trait_ty for #ty<T> where T: std::ops::Neg<Output = T> {
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

    pub fn trait_ty() -> syn::Type {
        parse_quote!(geo_traits::Reverse)
    }

    pub fn trait_fn() -> Ident {
        parse_quote!(rev)
    }
}

pub struct Antireverse;

impl Antireverse {
    fn impl_for(ty: Type, algebra: Algebra) -> ItemImpl {
        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();

        let fields = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            let rev = algebra.antirev(blade);
            if rev.is_positive() {
                quote! { #f: self.#f, }
            } else {
                quote! { #f: -self.#f, }
            }
        });

        parse_quote! {
            impl<T> #trait_ty for #ty<T>
            where
                T: std::ops::Neg<Output = T>
            {
                fn #trait_fn(self) -> Self {
                    #ty {
                        #(#fields)*
                    }
                }
            }
        }
    }

    pub fn trait_ty() -> syn::Type {
        parse_quote!(geo_traits::Antireverse)
    }

    pub fn trait_fn() -> Ident {
        parse_quote!(antirev)
    }
}

pub struct GradeProduct;

impl GradeProduct {
    pub fn impl_for(lhs: Type, rhs: Type, out: Type, algebra: Algebra) -> Option<ItemImpl> {
        if algebra.slim && out != Type::Grade(0) {
            return None;
        }

        let blades = Self::iter_blades(lhs, rhs, out, algebra).fold(
            HashMap::<Blade, Vec<(Blade, Blade, Blade)>>::new(),
            |mut map, (l, r, o)| {
                map.entry(o.unsigned()).or_default().push((l, r, o));
                map
            },
        );

        if blades.is_empty() {
            return None;
        }

        let fields = out.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            let mut sum = quote!();

            if let Some(vec) = blades.get(&blade) {
                for (l, r, o) in vec {
                    let lf = l.field(algebra);
                    let rf = r.field(algebra);
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

        let fn_attrs = fn_attrs();
        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();

        Some(parse_quote! {
            impl<T, U, V> #trait_ty<#lhs<T>, #rhs<U>> for #out<V>
            where
                T: std::ops::Mul<U, Output = V> + Copy,
                U: Copy,
                V: std::ops::Add<Output = V>
                    + std::ops::Sub<Output = V>
                    + std::ops::Neg<Output = V>
                    + num_traits::Zero,
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

    pub fn trait_ty() -> syn::Type {
        parse_quote!(geo_traits::GradeProduct)
    }

    pub fn trait_fn() -> Ident {
        parse_quote!(product)
    }

    fn iter_blades(
        lhs: Type,
        rhs: Type,
        out: Type,
        algebra: Algebra,
    ) -> impl Iterator<Item = (Blade, Blade, Blade)> {
        lhs.iter_blades_unsorted(algebra)
            .flat_map(move |lhs| {
                rhs.iter_blades_unsorted(algebra)
                    .map(move |rhs| (lhs, rhs, algebra.geo(lhs, rhs)))
            })
            .filter(move |(_l, _r, o)| out.contains(*o))
    }

    pub fn contains(lhs: Type, rhs: Type, out: Type, algebra: Algebra) -> bool {
        Self::iter_blades(lhs, rhs, out, algebra).any(|_| true)
    }
}

pub struct GradeAntiproduct;

impl GradeAntiproduct {
    pub fn trait_ty() -> syn::Type {
        parse_quote!(geo_traits::GradeAntiproduct)
    }

    pub fn trait_fn() -> Ident {
        parse_quote!(antiproduct)
    }

    pub fn impl_for(lhs: Type, rhs: Type, out: Type, algebra: Algebra) -> Option<ItemImpl> {
        if algebra.slim {
            return None;
        }

        let blades = lhs
            .iter_blades_unsorted(algebra)
            .flat_map(|lhs| {
                rhs.iter_blades_unsorted(algebra)
                    .map(move |rhs| (lhs, rhs, algebra.antigeo(lhs, rhs)))
            })
            .filter(|(_l, _r, o)| out.contains(*o))
            .fold(
                HashMap::<Blade, Vec<(Blade, Blade, Blade)>>::new(),
                |mut map, (l, r, o)| {
                    map.entry(o.unsigned()).or_default().push((l, r, o));
                    map
                },
            );

        if blades.is_empty() {
            return None;
        }

        let fields = out.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            let mut sum = quote!();

            if let Some(vec) = blades.get(&blade) {
                for (l, r, o) in vec {
                    let lf = l.field(algebra);
                    let rf = r.field(algebra);
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

        let fn_attrs = fn_attrs();
        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();

        Some(parse_quote! {
            impl<T, U, V> #trait_ty<#lhs<T>, #rhs<U>> for #out<V>
            where
                T: std::ops::Mul<U, Output = V> + Copy,
                U: Copy,
                V: std::ops::Add<Output = V>
                    + std::ops::Sub<Output = V>
                    + std::ops::Neg<Output = V>
                    + num_traits::Zero,
            {
                #fn_attrs
                fn #trait_fn(lhs: #lhs<T>, rhs: #rhs<U>) -> Self {
                    #out {
                        #(#fields)*
                    }
                }
            }
        })
    }
}

impl Type {
    pub fn define(self, algebra: Algebra) -> ItemStruct {
        let fields = self.iter_blades_sorted(algebra).map(|b| b.field(algebra));

        let attr = quote! {
            #[repr(C)]
            #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
        };

        parse_quote! {
            #attr
            pub struct #self <T> {
                #( pub #fields: T, )*
            }
        }
    }

    fn impl_grade_fns(self, algebra: Algebra) -> Option<ItemImpl> {
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
                let f = blade.field(algebra);
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
        Some(parse_quote! {
            impl<T> #self<T> {
                #(#fns)*
            }
        })
    }

    fn impl_new_fn(self, algebra: Algebra) -> ItemImpl {
        let params = self.iter_blades_sorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            quote!(#f: T)
        });

        let fields = self.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            quote! {
                #f: {
                    debug_assert!(#f.is_finite());
                    #f
                }
            }
        });

        parse_quote! {
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

    pub fn impl_map(self, algebra: Algebra) -> ItemImpl {
        let fields = self.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            quote! {
                #f: f(self.#f),
            }
        });
        parse_quote! {
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

    pub fn impl_from(self, algebra: Algebra) -> impl Iterator<Item = ItemImpl> {
        algebra.types().filter_map(move |target| {
            if self != target && target.contains_ty(self) {
                let fields = self.iter_blades_unsorted(algebra).map(|blade| {
                    let f = blade.field(algebra);
                    quote!(#f: value.#f,)
                });
                Some(parse_quote! {
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

    fn impl_complement(self, algebra: Algebra, op: Complement) -> ItemImpl {
        let ident = op.trait_ty();
        let fn_ident = op.trait_fn();
        let comp = self.complement(algebra);
        let fn_attrs = fn_attrs();
        let fields = self.iter_blades_unsorted(algebra).map(|blade| {
            let comp = op.call(algebra, blade);
            let cf = comp.field(algebra);
            let sf = blade.field(algebra);
            if comp.is_positive() {
                quote!(#cf: self.#sf,)
            } else {
                quote!(#cf: -self.#sf,)
            }
        });
        parse_quote! {
            impl<T> #ident for #self<T>
            where
                T: std::ops::Neg<Output = T>,
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

    fn impl_zero(self, algebra: Algebra) -> ItemImpl {
        let fields0 = self.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            quote! { #f: num_traits::Zero::zero(), }
        });
        let fields1 = self.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            quote! { self.#f.is_zero() }
        });
        parse_quote! {
            impl<T> num_traits::Zero for #self<T>
            where
                T: num_traits::Zero,
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

    fn impl_one(self, algebra: Algebra) -> Option<ItemImpl> {
        if !self.contains(Blade::scalar()) {
            return None;
        }

        let other_blades = self
            .iter_blades_unsorted(algebra)
            .any(|b| b != Blade::scalar());

        let where_clause = if other_blades {
            quote! {
                where
                    T: num_traits::One + num_traits::Zero,
                    #self<T>: std::ops::Mul<Output = #self<T>>,
            }
        } else {
            quote! {
                where
                    T: num_traits::One,
                    #self<T>: std::ops::Mul<Output = #self<T>>,
            }
        };

        let fields = self.iter_blades_unsorted(algebra).map(|b| {
            let field = b.field(algebra);
            let value = if b == Blade::scalar() {
                quote! { num_traits::One::one() }
            } else {
                quote! { num_traits::Zero::zero() }
            };
            quote! { #field: #value }
        });

        Some(parse_quote! {
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

    fn has_impl_sample_unit(self, algebra: Algebra) -> bool {
        matches!(self, Type::Grade(_))
            && !algebra.slim
            && !algebra.has_negative_bases()
            && self
                .iter_blades_unsorted(algebra)
                .any(|b| !algebra.dot(b, b).is_zero())
    }

    pub fn impl_sample_unit(self, algebra: Algebra) -> Option<ItemImpl> {
        if !self.has_impl_sample_unit(algebra) {
            return None;
        }

        let fields = self.iter_blades_unsorted(algebra).map(|b| b.field(algebra));

        let constructor = quote! {
            let v = #self { #( #fields ),* };
        };

        let fields = self.iter_blades_unsorted(algebra).map(|b| {
            let f = b.field(algebra);
            if algebra.dot(b, b).is_zero() {
                quote! {
                    let #f = T::zero();
                }
            } else {
                quote! {
                    let #f = rng.gen::<T>() * two - T::one();
                }
            }
        });

        let unit_ty = InverseOps::Unitize.trait_ty();
        let unit_fn = InverseOps::Unitize.trait_fn();
        let norm2_ty = NormOps::Norm2.trait_ty();
        let norm2_fn = NormOps::Norm2.trait_fn();
        Some(parse_quote! {
            impl<T> rand::distributions::Distribution<Unit<#self<T>>> for rand::distributions::Standard
            where
                T: num_traits::One
                    + num_traits::Zero
                    + std::ops::Add<Output = T>
                    + std::ops::Sub<Output = T>
                    + std::ops::Mul<Output = T>
                    + PartialOrd
                    + Copy,
                #self<T>: #norm2_ty<Output = Scalar<T>> + #unit_ty<Output = Unit<#self<T>>>,
                rand::distributions::Standard: rand::distributions::Distribution<T>,
            {
                #[inline]
                fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Unit<#self<T>> {
                    let mut i = 1u16;
                    while i != 0 {
                        let two = T::one() + T::one();
                        #( #fields )*
                        #constructor
                        if #norm2_ty::#norm2_fn(v).s <= T::one() {
                            return #unit_ty::#unit_fn(v);
                        }
                        i <<= 1;
                    }
                    panic!("unable to find unit value for {}", std::any::type_name::<Self>());
                }
            }
        })
    }

    fn test_sample_unit(self, algebra: Algebra) -> Option<ItemFn> {
        if !self.has_impl_sample_unit(algebra) {
            return None;
        }

        let fn_ident = Ident::new(
            &format!("test_sample_unit_{}", self.name_lowercase()),
            Span::mixed_site(),
        );

        let norm2_ty = NormOps::Norm2.trait_ty();
        let norm2_fn = NormOps::Norm2.trait_fn();

        Some(parse_quote! {
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
    pub fn trait_ty(self) -> syn::Type {
        match self {
            Self::Dual => parse_quote!(geo_traits::Dual),
            Self::LeftComp => parse_quote!(geo_traits::LeftComplement),
            Self::RightComp => parse_quote!(geo_traits::RightComplement),
        }
    }

    pub fn trait_fn(self) -> Ident {
        match self {
            Self::Dual => parse_quote!(dual),
            Self::LeftComp => parse_quote!(left_comp),
            Self::RightComp => parse_quote!(right_comp),
        }
    }
}

impl ToTokens for Type {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ident().to_tokens(tokens);
    }
}

impl Blade {
    pub fn field(self, algebga: Algebra) -> Ident {
        let mut output = String::new();
        for basis in algebga.iter_bases(self) {
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
    pub fn trait_ty(self) -> syn::Type {
        match self {
            ProductOp::Geo => parse_quote!(geo_traits::Geo),
            ProductOp::Wedge => parse_quote!(geo_traits::Wedge),
            ProductOp::Dot => parse_quote!(geo_traits::Dot),
            ProductOp::Antigeo => parse_quote!(geo_traits::Antigeo),
            ProductOp::Antidot => parse_quote!(geo_traits::Antidot),
            ProductOp::Antiwedge => parse_quote!(geo_traits::Antiwedge),
            ProductOp::Mul => parse_quote!(std::ops::Mul),
            ProductOp::Commutator => parse_quote!(geo_traits::Commutator),
        }
    }

    pub fn trait_fn(self) -> Ident {
        match self {
            ProductOp::Geo => parse_quote!(geo),
            ProductOp::Wedge => parse_quote!(wedge),
            ProductOp::Dot => parse_quote!(dot),
            ProductOp::Antigeo => parse_quote!(antigeo),
            ProductOp::Antidot => parse_quote!(antidot),
            ProductOp::Antiwedge => parse_quote!(antiwedge),
            ProductOp::Mul => parse_quote!(mul),
            ProductOp::Commutator => parse_quote!(com),
        }
    }

    fn impl_for(self, algebra: Algebra, lhs: Type, rhs: Type) -> Option<ItemImpl> {
        let i = Type::Grade(algebra.bases.len() as u32);
        if algebra.slim
            && rhs != Type::Grade(1)
            && lhs != Type::Grade(0)
            && rhs != Type::Grade(0)
            && rhs != i
        {
            return None;
        }

        let op = self.trait_ty();
        let op_fn = self.trait_fn();
        let output = self.output(algebra, lhs, rhs)?;

        if !algebra.contains(output) {
            return None;
        }

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
            let ident = blade.field(algebra);

            let mut sum = quote!();

            if let Some(vec) = blades.get(&blade) {
                for (l, r, o) in vec {
                    let lf = l.field(algebra);
                    let rf = r.field(algebra);
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

        Some(parse_quote! {
            impl<T, U, V> #op<#rhs<U>> for #lhs<T>
            where
                T: std::ops::Mul<U, Output = V> + Copy,
                U: Copy,
                V: std::ops::Add<Output = V>
                    + std::ops::Sub<Output = V>
                    + std::ops::Neg<Output = V>
                    + num_traits::Zero,
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
    fn impl_for(self, algebra: Algebra, lhs: Type, rhs: Type) -> Option<ItemImpl> {
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
            let field = blade.field(algebra);
            let mut sum = quote!();
            if let Some(vec) = blades.get(&blade) {
                for (side, blade) in vec {
                    let f = blade.field(algebra);
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

        // Allow more generic type flexibility for operations within a type
        if lhs == rhs {
            Some(parse_quote! {
                impl<T, U, V> #trait_ty<#rhs<U>> for #lhs<T>
                where
                    T: #trait_ty<U, Output = V>
                {
                    type Output = #output<V>;
                    #fn_attrs
                    fn #trait_fn(self, rhs: #rhs<U>) -> Self::Output {
                        #output {
                            #(#fields)*
                        }
                    }
                }
            })
        } else {
            Some(parse_quote! {
                impl<T> #trait_ty<#rhs<T>> for #lhs<T>
                where
                    T: std::ops::Add<Output = T>
                        + std::ops::Sub<Output = T>
                        + std::ops::Neg<Output = T>
                        + num_traits::Zero,
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
    }

    pub fn trait_ty(self) -> syn::Type {
        match self {
            SumOp::Add => parse_quote! { std::ops::Add },
            SumOp::Sub => parse_quote! { std::ops::Sub },
        }
    }

    pub fn trait_fn(self) -> Ident {
        match self {
            SumOp::Add => parse_quote! { add },
            SumOp::Sub => parse_quote! { sub },
        }
    }
}

impl ScalarOps {
    pub fn impl_for_scalar(self, ty: Type, algebra: Algebra) -> Option<Vec<ItemImpl>> {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();

        let fn_attrs = quote!(
            #[inline]
            #[allow(clippy::suspicious_arithmetic_impl)]
        );

        if self == Self::Div {
            let inv_ty = InverseOps::Inverse.trait_ty();
            let inv_fn = InverseOps::Inverse.trait_fn();
            Some(vec![
                parse_quote! {
                    impl<U, V> #trait_ty<#ty<U>> for f32
                    where
                        #ty<U>: #inv_ty<Output = #ty<U>>,
                        Scalar<f32>: std::ops::Mul<#ty<U>, Output = #ty<V>>,
                        U: Copy,
                    {
                        type Output = #ty<V>;
                        #fn_attrs
                        fn #trait_fn(self, rhs: #ty<U>) -> Self::Output {
                            Scalar::new(self) * #inv_ty::#inv_fn(rhs)
                        }
                    }
                },
                parse_quote! {
                    impl<U, V> #trait_ty<#ty<U>> for f64
                    where
                        #ty<U>: #inv_ty<Output = #ty<U>>,
                        Scalar<f64>: std::ops::Mul<#ty<U>, Output = #ty<V>>,
                        U: Copy,
                    {
                        type Output = #ty<V>;
                        #fn_attrs
                        fn #trait_fn(self, rhs: #ty<U>) -> Self::Output {
                            Scalar::new(self) * #inv_ty::#inv_fn(rhs)
                        }
                    }
                },
                parse_quote! {
                    impl<T> #trait_ty<f32> for #ty<T>
                    where
                        #ty<T>: std::ops::Div<Scalar<f32>, Output = #ty<T>>,
                    {
                        type Output = #ty<T>;
                        #fn_attrs
                        fn #trait_fn(self, rhs: f32) -> Self::Output {
                            self / Scalar { s: rhs }
                        }
                    }
                },
                parse_quote! {
                    impl<T> #trait_ty<f64> for #ty<T>
                    where
                        #ty<T>: std::ops::Div<Scalar<f64>, Output = #ty<T>>,
                    {
                        type Output = #ty<T>;
                        #fn_attrs
                        fn #trait_fn(self, rhs: f64) -> Self::Output {
                            self / Scalar { s: rhs }
                        }
                    }
                },
            ])
        } else {
            let fields = ty.iter_blades_unsorted(algebra).map(|blade| {
                let f = blade.field(algebra);
                quote! { #f: #trait_ty::#trait_fn(self.#f, rhs), }
            });
            let fields1 = fields.clone();
            Some(vec![
                parse_quote! {
                    impl<T> #trait_ty<#ty<T>> for f32
                    where
                        #ty<T>: #trait_ty<f32, Output = #ty<T>>,
                        T: Copy,
                    {
                        type Output = #ty<T>;
                        #fn_attrs
                        fn #trait_fn(self, rhs: #ty<T>) -> Self::Output {
                            #trait_ty::#trait_fn(rhs, self)
                        }
                    }
                },
                parse_quote! {
                    impl<T> #trait_ty<#ty<T>> for f64
                    where
                        #ty<T>: #trait_ty<f64, Output = #ty<T>>,
                        T: Copy,
                    {
                        type Output = #ty<T>;
                        #fn_attrs
                        fn #trait_fn(self, rhs: #ty<T>) -> Self::Output {
                            #trait_ty::#trait_fn(rhs, self)
                        }
                    }
                },
                parse_quote! {
                    impl<T, U> #trait_ty<f32> for #ty<T>
                    where
                        T: #trait_ty<f32, Output = U>,
                    {
                        type Output = #ty<U>;
                        #fn_attrs
                        fn #trait_fn(self, rhs: f32) -> Self::Output {
                            #ty {
                                #(#fields)*
                            }
                        }
                    }
                },
                parse_quote! {
                    impl<T, U> #trait_ty<f64> for #ty<T>
                    where
                        T: #trait_ty<f64, Output = U>,
                    {
                        type Output = #ty<U>;
                        #fn_attrs
                        fn #trait_fn(self, rhs: f64) -> Self::Output {
                            #ty {
                                #(#fields1)*
                            }
                        }
                    }
                },
            ])
        }
    }

    pub fn trait_ty(self) -> syn::Type {
        match self {
            Self::Mul => parse_quote!(std::ops::Mul),
            Self::Div => parse_quote!(std::ops::Div),
        }
    }

    pub fn trait_fn(self) -> syn::Type {
        match self {
            Self::Mul => parse_quote!(mul),
            Self::Div => parse_quote!(div),
        }
    }
}

impl ScalarAssignOps {
    pub fn impl_for(self, ty: Type, algebra: Algebra) -> ItemImpl {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let fn_attrs = fn_attrs();
        let fields = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            quote! { #trait_ty::#trait_fn(&mut self.#f, rhs); }
        });
        parse_quote! {
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

    pub fn trait_ty(self) -> syn::Type {
        match self {
            Self::Mul => parse_quote!(std::ops::MulAssign),
            Self::Div => parse_quote!(std::ops::DivAssign),
        }
    }

    pub fn trait_fn(self) -> syn::Type {
        match self {
            Self::Mul => parse_quote!(mul_assign),
            Self::Div => parse_quote!(div_assign),
        }
    }
}

impl SumAssignOps {
    pub fn impl_for(self, ty: Type, algebra: Algebra) -> ItemImpl {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let fn_attrs = fn_attrs();
        let fields = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            quote! { #trait_ty::#trait_fn(&mut self.#f, rhs.#f); }
        });
        parse_quote! {
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

    pub fn trait_ty(self) -> syn::Type {
        match self {
            Self::AddAssign => parse_quote!(std::ops::AddAssign),
            Self::SubAssign => parse_quote!(std::ops::SubAssign),
        }
    }

    pub fn trait_fn(self) -> syn::Type {
        match self {
            Self::AddAssign => parse_quote!(add_assign),
            Self::SubAssign => parse_quote!(sub_assign),
        }
    }
}

impl FloatConversion {
    pub fn impl_for(self, ty: Type, algebra: Algebra) -> ItemImpl {
        let from = self.type_from();
        let to = self.type_to();
        let fn_ident = self.fn_ident();
        let fields1 = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            quote! {
                #f: self.#f as #to
            }
        });
        let fn_attrs = fn_attrs();
        parse_quote! {
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

    fn type_from(self) -> syn::Type {
        match self {
            Self::ToF64 => parse_quote!(f32),
            Self::ToF32 => parse_quote!(f64),
        }
    }

    fn type_to(self) -> syn::Type {
        match self {
            Self::ToF32 => parse_quote!(f32),
            Self::ToF64 => parse_quote!(f64),
        }
    }

    fn fn_ident(self) -> Ident {
        match self {
            Self::ToF32 => parse_quote!(to_f32),
            Self::ToF64 => parse_quote!(to_f64),
        }
    }
}

impl NormOps {
    pub fn impl_for(self, ty: Type, _algebra: Algebra) -> ItemImpl {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let fn_attrs = fn_attrs();

        let grade_prod_ty = GradeProduct::trait_ty();
        let rev_ty = Reverse::trait_ty();
        let rev_fn = Reverse::trait_fn();

        match self {
            NormOps::Norm2 => {
                parse_quote! {
                    impl<T> #trait_ty for #ty<T>
                    where
                        Scalar<T>: #grade_prod_ty<#ty<T>, #ty<T>, Output = Scalar<T>>,
                        #ty<T>: #rev_ty<Output = #ty<T>> + Copy,
                        T: std::ops::Mul<Output = T>,
                    {
                        type Output = Scalar<T>;
                        #fn_attrs
                        fn #trait_fn(self) -> Self::Output {
                            use #grade_prod_ty;
                            Scalar::product(self, #rev_ty::#rev_fn(self))
                        }
                    }
                }
            }
            NormOps::Norm => {
                parse_quote! {
                    impl<T> #trait_ty for #ty<T>
                    where
                        Scalar<T>: #grade_prod_ty<#ty<T>, #ty<T>, Output = Scalar<T>>,
                        #ty<T>: #rev_ty<Output = #ty<T>> + Copy,
                        T: std::ops::Mul<Output = T>,
                        T: num_sqrt::Sqrt<Output = T>,
                    {
                        type Output = Scalar<T>;
                        #fn_attrs
                        fn #trait_fn(self) -> Self::Output {
                            use #grade_prod_ty;
                            let scalar = Scalar::product(self, #rev_ty::#rev_fn(self));
                            num_sqrt::Sqrt::sqrt(scalar)
                        }
                    }
                }
            }
        }
    }

    pub fn trait_ty(self) -> syn::Type {
        match self {
            NormOps::Norm => parse_quote!(geo_traits::Norm),
            NormOps::Norm2 => parse_quote!(geo_traits::Norm2),
        }
    }

    pub fn trait_fn(self) -> Ident {
        match self {
            NormOps::Norm => parse_quote!(norm),
            NormOps::Norm2 => parse_quote!(norm2),
        }
    }
}

pub struct Sandwich;

impl Sandwich {
    pub fn impl_for(lhs: Type, rhs: Type, algebra: Algebra) -> Option<ItemImpl> {
        if InverseOps::Inverse.inapplicable(lhs, algebra) || algebra.slim {
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
        Some(parse_quote! {
            impl<T, U, V, W> #trait_ty<#rhs<U>> for #lhs<T>
            where
                #lhs<T>: #geo_ty<#rhs<U>, Output = #intermediate<V>>
                    + #inv_ty<Output = #lhs<T>>
                    + Copy,
                #rhs<W>: #grade_prod_ty<#intermediate<V>, #lhs<T>, Output = #rhs<W>>,
                V: std::ops::Mul<T, Output = W>,
            {
                type Output = #rhs<W>;
                #fn_attrs
                fn #trait_fn(self, rhs: #rhs<U>) -> Self::Output {
                    let intermediate: #intermediate<V> = #geo_ty::#geo_fn(self, rhs);
                    #rhs::#grade_prod_fn(intermediate, #inv_ty::#inv_fn(self))
                }
            }
        })
    }

    pub fn trait_ty() -> syn::Type {
        parse_quote!(geo_traits::Sandwich)
    }

    pub fn trait_fn() -> Ident {
        parse_quote!(sandwich)
    }
}

pub struct Antisandwich;

impl Antisandwich {
    pub fn impl_for(lhs: Type, rhs: Type, algebra: Algebra) -> Option<ItemImpl> {
        if InverseOps::Inverse.inapplicable(lhs.complement(algebra), algebra) || algebra.slim {
            return None;
        }

        let intermediate = ProductOp::Geo.output(algebra, lhs, rhs)?;
        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();
        let fn_attrs = fn_attrs();
        let geo_ty = ProductOp::Antigeo.trait_ty();
        let geo_fn = ProductOp::Antigeo.trait_fn();
        let inv_ty = InverseOps::Inverse.trait_ty();
        let inv_fn = InverseOps::Inverse.trait_fn();
        let grade_prod_ty = GradeProduct::trait_ty();
        let grade_prod_fn = GradeProduct::trait_fn();
        Some(parse_quote! {
            impl<T, U, V, W> #trait_ty<#rhs<U>> for #lhs<T>
            where
                #lhs<T>: #geo_ty<#rhs<U>, Output = #intermediate<V>>
                    + #inv_ty<Output = #lhs<T>>
                    + Copy,
                #rhs<W>: #grade_prod_ty<#intermediate<V>, #lhs<T>, Output = #rhs<W>>,
                V: std::ops::Mul<T, Output = W>,
            {
                type Output = #rhs<W>;
                #fn_attrs
                fn #trait_fn(self, rhs: #rhs<U>) -> Self::Output {
                    let intermediate: #intermediate<V> = #geo_ty::#geo_fn(self, rhs);
                    #rhs::<W>::#grade_prod_fn(intermediate, #inv_ty::#inv_fn(self))
                }
            }
        })
    }

    pub fn trait_ty() -> syn::Type {
        parse_quote!(geo_traits::Antisandwich)
    }

    pub fn trait_fn() -> Ident {
        parse_quote!(antisandwich)
    }
}

impl InverseOps {
    pub fn trait_ty(self) -> syn::Type {
        match self {
            InverseOps::Inverse => parse_quote!(geo_traits::Inverse),
            InverseOps::Unitize => parse_quote!(geo_traits::Unitize),
        }
    }

    pub fn trait_fn(self) -> Ident {
        match self {
            InverseOps::Inverse => parse_quote!(inv),
            InverseOps::Unitize => parse_quote!(unit),
        }
    }

    /// No implementation if there's only one blade, or mv which are not easily inverted
    pub fn inapplicable(self, ty: Type, algebra: Algebra) -> bool {
        algebra.has_negative_bases()
            || ty == Type::Mv
            || ProductOp::Dot.output(algebra, ty, ty).is_none()
    }

    pub fn impl_for(self, ty: Type, algebra: Algebra) -> Option<ItemImpl> {
        if self.inapplicable(ty, algebra) {
            return None;
        }

        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();

        let fn_attrs = fn_attrs();
        let fn_attrs = quote! {
            #[track_caller]
            #fn_attrs
        };

        let inv_ty = InverseOps::Inverse.trait_ty();
        let inv_fn = InverseOps::Inverse.trait_fn();

        let rev_ty = Reverse::trait_ty();
        let rev_fn = Reverse::trait_fn();

        let norm2_ty = NormOps::Norm2.trait_ty();
        let norm2_fn = NormOps::Norm2.trait_fn();

        let norm_ty = NormOps::Norm.trait_ty();
        let norm_fn = NormOps::Norm.trait_fn();

        Some(match (self, ty) {
            (InverseOps::Inverse, Type::Grade(0)) => {
                parse_quote! {
                    impl<T> #trait_ty for #ty<T>
                    where
                        T: num_traits::One
                            + std::ops::Div<Output = T>,
                    {
                        type Output = #ty<T>;
                        #fn_attrs
                        fn #trait_fn(self) -> Self::Output {
                            Scalar { s: T::one() / self.s }
                        }
                    }
                }
            }
            (InverseOps::Inverse, _) => {
                parse_quote! {
                    impl<T> #trait_ty for #ty<T>
                    where
                        #ty<T>: #norm2_ty<Output = Scalar<T>>
                            + #rev_ty<Output = #ty<T>>
                            + std::ops::Mul<Scalar<T>, Output = #ty<T>>
                            + Copy,
                        Scalar<T>: #inv_ty<Output = Scalar<T>>,
                    {
                        type Output = #ty<T>;
                        #fn_attrs
                        fn #trait_fn(self) -> Self::Output {
                            #rev_ty::#rev_fn(self) * #inv_ty::#inv_fn(#norm2_ty::#norm2_fn(self))
                        }
                    }
                }
            }
            (InverseOps::Unitize, Type::Grade(0)) => {
                parse_quote! {
                    impl<T> #trait_ty for #ty<T>
                    where
                        T: num_traits::One,
                    {
                        type Output = Unit<#ty<T>>;
                        #fn_attrs
                        fn #trait_fn(self) -> Self::Output {
                            Unit(Scalar { s: T::one() })
                        }
                    }
                }
            }
            (InverseOps::Unitize, _) => {
                parse_quote! {
                    impl<T> #trait_ty for #ty<T>
                    where
                        #ty<T>: #norm_ty<Output = Scalar<T>>
                            + std::ops::Mul<Scalar<T>, Output = #ty<T>>
                            + Copy,
                        Scalar<T>: #inv_ty<Output = Scalar<T>>,
                    {
                        type Output = Unit<#ty<T>>;
                        #fn_attrs
                        fn #trait_fn(self) -> Self::Output {
                            Unit(self * #inv_ty::#inv_fn(#norm_ty::#norm_fn(self)))
                        }
                    }
                }
            }
        })
    }

    pub fn tests(self, ty: Type, algebra: Algebra) -> Option<syn::ItemFn> {
        if self.inapplicable(ty, algebra) || algebra.slim {
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
            let f = blade.field(algebra);
            quote! { #f: rng.gen::<f64>(), }
        });

        let geo_ty = ProductOp::Geo.trait_ty();
        let geo_fn = ProductOp::Geo.trait_fn();

        let expr = match self {
            InverseOps::Inverse => quote! {
                let inv = value.inv();
                let product = #geo_ty::#geo_fn(value, inv);
                let remainder = (product - Scalar { s: 1. }).norm2();
                dbg!(value, inv, product);
                assert!(remainder.s < 1e-10);
            },
            InverseOps::Unitize => quote! {
                let unit = value.unit().value();
                let product = #geo_ty::#geo_fn(unit, unit.rev());
                let remainder = (product - Scalar { s: 1. }).norm2();
                dbg!(value, unit, product);
                assert!(remainder.s < 1e-10);
            },
        };

        Some(parse_quote! {
            #[test]
            fn #fn_ident() {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let value = #ty {
                    #(#fields)*
                };
                #expr
            }
        })
    }
}

pub struct Unit;

impl Unit {
    pub fn define(algebra: Algebra) -> Vec<syn::Item> {
        let unitize_ty = InverseOps::Unitize.trait_ty();
        let unitize_fn = InverseOps::Unitize.trait_fn();
        let inverse_ty = InverseOps::Inverse.trait_ty();
        let inverse_fn = InverseOps::Inverse.trait_fn();
        let reverse_ty = Reverse::trait_ty();
        let item_struct = parse_quote! {
            #[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
            pub struct Unit<T>(T);
        };
        let item_impl = parse_quote! {
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
        let unitize_impl = parse_quote! {
            impl<T> #unitize_ty for Unit<T> where T: #unitize_ty {
                type Output = Self;
                #[inline]
                fn #unitize_fn(self) -> Self::Output {
                    self
                }
            }
        };
        let inverse_impl = parse_quote! {
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
        let sandwich_impl = parse_quote! {
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
        let antigeo_ty = ProductOp::Antigeo.trait_ty();
        let antigeo_fn = ProductOp::Antigeo.trait_fn();
        let antisandwich_ty = Antisandwich::trait_ty();
        let antisandwich_fn = Antisandwich::trait_fn();
        let antisandwich_impl = parse_quote! {
            impl<Lhs, Rhs, Int> #antisandwich_ty<Rhs> for Unit<Lhs>
            where
                Lhs: #antigeo_ty<Rhs, Output = Int> + #rev_ty<Output = Lhs> + Copy,
                Rhs: #grade_prod_ty<Int, Lhs, Output = Rhs>,
            {
                type Output = Rhs;
                #[inline]
                fn #antisandwich_fn(self, rhs: Rhs) -> Self::Output {
                    let int = #antigeo_ty::#antigeo_fn(self.value(), rhs);
                    Rhs::product(int, self.value().rev())
                }
            }
        };

        use syn::Item::*;

        let operator_overloads = algebra
            .types()
            .flat_map(|ty| Overload::iter(algebra).filter_map(move |op| op.impl_for_unit(ty)))
            .map(Impl);

        IntoIterator::into_iter([
            Struct(item_struct),
            Impl(item_impl),
            Impl(unitize_impl),
            Impl(inverse_impl),
            Impl(sandwich_impl),
            Impl(antisandwich_impl),
        ])
        .chain(ProductOp::iter_all(algebra).map(|op| {
            let trait_ty = op.trait_ty();
            let trait_fn = op.trait_fn();
            parse_quote! {
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

pub struct Sqrt;

impl Sqrt {
    pub fn impl_for(ty: Type, _algebra: Algebra) -> Option<syn::ItemImpl> {
        let sqrt_ty = Self::trait_ty();
        let sqrt_fn = Self::trait_fn();
        let fn_attrs = fn_attrs();
        match ty {
            Type::Grade(0) => Some(parse_quote! {
                impl<T, U> #sqrt_ty for #ty<T>
                where
                    T: #sqrt_ty<Output = U>,
                {
                    type Output = Scalar<U>;
                    #fn_attrs
                    fn #sqrt_fn(self) -> Self::Output {
                        Scalar {
                            s: num_sqrt::Sqrt::sqrt(self.s)
                        }
                    }
                }
            }),
            // TODO pin these down for Bivectors and Motors
            //    does it require a unitized type?
            //    can we just add Scalar { s: self.norm2() } and then unitize?
            _ => None,
        }
    }

    pub fn trait_ty() -> syn::Type {
        parse_quote!(num_sqrt::Sqrt)
    }

    pub fn trait_fn() -> Ident {
        parse_quote!(sqrt)
    }
}

pub struct Div;

impl Div {
    pub fn impl_for(lhs: Type, rhs: Type, algebra: Algebra) -> Option<ItemImpl> {
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

        Some(parse_quote! {
            impl<T, U, V> std::ops::Div<#rhs<U>> for #lhs<T>
            where
                #lhs<T>: std::ops::Mul<#rhs<U>, Output = #output<V>>,
                #rhs<U>: #inv_ty<Output = #rhs<U>>,
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
pub enum Overload {
    And,
    Or,
    Xor,
    Shr,
}

impl Overload {
    pub fn iter(algebra: Algebra) -> impl Iterator<Item = Self> {
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

    pub fn impl_for(self, lhs: Type, rhs: Type, algebra: Algebra) -> Option<ItemImpl> {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let inner = self.inner_op();
        let inner_ty = inner.trait_ty();
        let inner_fn = inner.trait_fn();
        let output = inner.output(algebra, lhs, rhs)?;
        if !algebra.contains(output) {
            return None;
        }
        Some(parse_quote! {
            impl<T, U, V> #trait_ty<#rhs<U>> for #lhs<T>
            where
                #lhs<T>: #inner_ty<#rhs<U>, Output = #output<V>>,
            {
                type Output = #output<V>;
                #[inline]
                fn #trait_fn(self, rhs: #rhs<U>) -> Self::Output {
                    #inner_ty::#inner_fn(self, rhs)
                }
            }
        })
    }

    pub fn impl_for_unit(self, lhs: Type) -> Option<ItemImpl> {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let inner = self.inner_op();
        let inner_ty = inner.trait_ty();
        let inner_fn = inner.trait_fn();
        Some(parse_quote! {
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

    pub fn trait_ty(self) -> syn::Type {
        match self {
            Overload::And => parse_quote!(std::ops::BitAnd),
            Overload::Or => parse_quote!(std::ops::BitOr),
            Overload::Xor => parse_quote!(std::ops::BitXor),
            Overload::Shr => parse_quote!(std::ops::Shr),
        }
    }

    pub fn trait_fn(self) -> syn::Type {
        match self {
            Overload::And => parse_quote!(bitand),
            Overload::Or => parse_quote!(bitor),
            Overload::Xor => parse_quote!(bitxor),
            Overload::Shr => parse_quote!(shr),
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
    fn trait_ty(self) -> syn::Type {
        match self {
            Self::Product(op) => op.trait_ty(),
            Self::Sandwich => Sandwich::trait_ty(),
        }
    }
    fn trait_fn(self) -> Ident {
        match self {
            Self::Product(op) => op.trait_fn(),
            Self::Sandwich => Sandwich::trait_fn(),
        }
    }
    fn output(self, algebra: Algebra, lhs: Type, rhs: Type) -> Option<Type> {
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
            Self {
                bases: &[
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
                ],
                slim: false,
            }
        }
    }

    #[test]
    #[ignore]
    fn write_to_file() {
        let path = "../output.rs";

        let bases = Box::new([
            Basis::pos('x'),
            Basis::pos('y'),
            Basis::pos('z'),
            Basis::pos('t'),
            Basis::pos('u'),
            Basis::pos('v'),
            Basis::pos('w'),
        ]);
        let algebra = Algebra {
            bases: Box::leak(bases),
            slim: true,
        };

        let output = format!("{}", algebra.define());
        std::fs::write(path, output).unwrap();
    }

    #[test]
    fn define_type() {
        let tokens = Type::Grade(2)
            .define(Algebra::ga3())
            .to_token_stream()
            .to_string();
        let expected = quote! {
            #[repr(C)]
            #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
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
        let a = Algebra {
            bases: &[Basis {
                char: '1',
                sqr: Square::Pos,
            }],
            slim: true,
        };
        assert_eq!("e1", a.blades().nth(1).unwrap().field(a).to_string());
    }
}

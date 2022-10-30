use crate::algebra::*;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use std::collections::HashMap;
use syn::{parse_quote, ItemImpl, ItemStruct, ItemTrait};

// TODO fn new for types to test for finiteness
// TODO build tests from https://docslib.org/doc/2164884/introduction-to-geometric-algebra-lecture-iv

impl Algebra {
    pub fn define(self) -> TokenStream {
        let traits = ProductOp::define_all()
            .chain(Complement::iter(self).map(Complement::define))
            .chain([Reverse::define(), Antireverse::define()])
            .chain([GradeProduct::define()])
            .chain([GradeAntiproduct::define()])
            .chain(NormOps::iter().map(NormOps::define))
            .chain([Sandwich::define(), Antisandwich::define()])
            .chain(InverseOps::iter().map(InverseOps::define));

        let structs = self.types().map(move |ty| ty.define(self));

        let impl_new_fn = self.types().map(|ty| ty.impl_new_fn(self));
        let impl_map_fn = self.types().map(|ty| ty.impl_map(self));
        let impl_grade_fns = self.types().filter_map(|ty| ty.impl_grade_fns(self));
        let impl_from = self.types().flat_map(|ty| ty.impl_from(self));

        let impl_zero = self.types().map(|ty| ty.impl_zero(self));
        let impl_one = self.types().filter_map(|ty| ty.impl_one(self));

        let impl_product_ops = ProductOp::iter_all()
            .flat_map(|op| self.types().map(move |ty| (op, ty)))
            .flat_map(|(op, lhs)| self.types().map(move |rhs| (op, lhs, rhs)))
            .filter_map(|(op, lhs, rhs)| op.impl_for(self, lhs, rhs));

        let div_ops = self
            .type_tuples()
            .filter_map(|(lhs, rhs)| Div::impl_for(lhs, rhs, self));

        let impl_sum_ops = SumOp::iter()
            .flat_map(|op| self.types().map(move |lhs| (op, lhs)))
            .flat_map(|(op, lhs)| self.types().map(move |rhs| (op, lhs, rhs)))
            .filter_map(|(op, lhs, rhs)| op.impl_for(self, lhs, rhs));

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

        let unit_items = Unit::define();

        let inverse_ops = InverseOps::iter()
            .flat_map(|op| self.types().filter_map(move |ty| op.impl_for(ty, self)));

        let impl_sqrt = self.types().map(move |ty| Sqrt::impl_for(ty, self));

        let test_unit_inv =
            InverseOps::iter().flat_map(|op| self.types().filter_map(move |ty| op.tests(ty, self)));

        let dynamic_types = self.dynamic_types();

        quote!(
            #(#traits)*
            #(#structs)*
            #(#impl_new_fn)*
            #(#impl_map_fn)*
            #(#impl_grade_fns)*
            #(#impl_from)*
            #(#impl_zero)*
            #(#impl_one)*
            #(#impl_product_ops)*
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
            #(#unit_items)*

            #[cfg(test)]
            mod tests {
                use super::*;
                #(#test_unit_inv)*
            }

            #dynamic_types
        )
    }
}

fn fn_attrs() -> TokenStream {
    quote! {
        #[inline]
        #[allow(unused_variables)]
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
            impl<T, U> std::ops::Neg for #ty<T>
            where
                T: std::ops::Neg<Output = U>,
            {
                type Output = #ty<U>;
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
    fn define() -> ItemTrait {
        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();
        parse_quote! {
            pub trait #trait_ty {
                fn #trait_fn(self) -> Self;
            }
        }
    }

    fn impl_for(ty: Type, algebra: Algebra) -> ItemImpl {
        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();

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
                fn #trait_fn(self) -> Self {
                    #ty {
                        #(#fields)*
                    }
                }
            }
        }
    }

    pub fn trait_ty() -> syn::Type {
        parse_quote!(Reverse)
    }

    pub fn trait_fn() -> Ident {
        parse_quote!(rev)
    }
}

pub struct Antireverse;

impl Antireverse {
    fn define() -> ItemTrait {
        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();
        parse_quote! {
            pub trait #trait_ty {
                fn #trait_fn(self) -> Self;
            }
        }
    }

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
            impl<T> #trait_ty for #ty<T> where T: std::ops::Neg<Output = T> {
                fn #trait_fn(self) -> Self {
                    #ty {
                        #(#fields)*
                    }
                }
            }
        }
    }

    pub fn trait_ty() -> syn::Type {
        parse_quote!(Antireverse)
    }

    pub fn trait_fn() -> Ident {
        parse_quote!(antirev)
    }
}

pub struct GradeProduct;

impl GradeProduct {
    pub fn define() -> ItemTrait {
        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();
        parse_quote! {
            pub trait #trait_ty<Lhs, Rhs> {
                type Output;
                fn #trait_fn(lhs: Lhs, rhs: Rhs) -> Self::Output;
            }
        }
    }

    pub fn impl_for(lhs: Type, rhs: Type, out: Type, algebra: Algebra) -> Option<ItemImpl> {
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
        parse_quote!(GradeProduct)
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
    pub fn define() -> ItemTrait {
        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();
        parse_quote! {
            pub trait #trait_ty<Lhs, Rhs> {
                fn #trait_fn(lhs: Lhs, rhs: Rhs) -> Self;
            }
        }
    }

    pub fn trait_ty() -> syn::Type {
        parse_quote!(GradeAntiproduct)
    }

    pub fn trait_fn() -> Ident {
        parse_quote!(antiproduct)
    }

    pub fn impl_for(lhs: Type, rhs: Type, out: Type, algebra: Algebra) -> Option<ItemImpl> {
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

        let attr = if self.single_blade(algebra) {
            quote! {
                #[repr(C)]
                #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
            }
        } else {
            quote! {
                #[repr(C)]
                #[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
            }
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
                #[inline]
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
        if self != Type::Grade(0) {
            return None;
        }

        let f = self
            .iter_blades_unsorted(algebra)
            .map(|blade| blade.field(algebra))
            .next()
            .unwrap();

        Some(parse_quote! {
            impl<T> num_traits::One for #self<T>
            where
                T: num_traits::One,
                #self<T>: std::ops::Mul<Output = #self<T>>,
            {
                fn one() -> Self {
                    #self {
                        #f: num_traits::One::one(),
                    }
                }
            }
        })
    }

    pub fn fn_ident(self) -> Ident {
        Ident::new(self.name_lowercase(), Span::mixed_site())
    }
}

impl Complement {
    pub fn trait_ty(self) -> Ident {
        match self {
            Self::Dual => parse_quote!(Dual),
            Self::LeftComp => parse_quote!(LeftComp),
            Self::RightComp => parse_quote!(RightComp),
        }
    }

    pub fn trait_fn(self) -> Ident {
        match self {
            Self::Dual => parse_quote!(dual),
            Self::LeftComp => parse_quote!(left_comp),
            Self::RightComp => parse_quote!(right_comp),
        }
    }

    fn define(self) -> ItemTrait {
        let ident = self.trait_ty();
        let fn_ident = self.trait_fn();
        parse_quote! {
            pub trait #ident {
                type Output;
                fn #fn_ident(self) -> Self::Output;
            }
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
        }
        Ident::new(&output, Span::mixed_site())
    }
}

impl ProductOp {
    fn define_all() -> impl Iterator<Item = ItemTrait> {
        Self::iter_local().map(Self::define)
    }

    fn define(self) -> ItemTrait {
        let op = self.trait_ty();
        let op_fn = self.trait_fn();
        parse_quote! {
            pub trait #op<Rhs> {
                type Output;
                fn #op_fn(self, rhs: Rhs) -> Self::Output;
            }
        }
    }

    pub fn trait_ty(self) -> syn::Type {
        match self {
            ProductOp::Geo => parse_quote!(Geo),
            ProductOp::Wedge => parse_quote!(Wedge),
            ProductOp::Dot => parse_quote!(Dot),
            ProductOp::Antigeo => parse_quote!(Antigeo),
            ProductOp::Antidot => parse_quote!(Antidot),
            ProductOp::Antiwedge => parse_quote!(Antiwedge),
            ProductOp::Mul => parse_quote!(std::ops::Mul),
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
        }
    }

    fn impl_for(self, algebra: Algebra, lhs: Type, rhs: Type) -> Option<ItemImpl> {
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

        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let output = Self::sum(algebra, lhs, rhs)?;

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
    pub fn impl_for_scalar(self, ty: Type, algebra: Algebra) -> Option<[ItemImpl; 4]> {
        if self == Self::Div {
            return None;
        }

        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let fn_attrs = fn_attrs();
        let fields = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            quote! { #f: #trait_ty::<U>::#trait_fn(self, rhs.#f), }
        });
        let fields1 = fields.clone();
        let fields2 = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            quote! { #f: #trait_ty::#trait_fn(self.#f, rhs), }
        });
        let fields3 = fields2.clone();
        Some([
            parse_quote! {
                impl<U, V> #trait_ty<#ty<U>> for f32
                where
                    f32: #trait_ty<U, Output = V>,
                    U: Copy,
                {
                    type Output = #ty<V>;
                    #fn_attrs
                    fn #trait_fn(self, rhs: #ty<U>) -> Self::Output {
                        #ty {
                            #(#fields)*
                        }
                    }
                }
            },
            parse_quote! {
                impl<U, V> #trait_ty<#ty<U>> for f64
                where
                    f64: #trait_ty<U, Output = V>,
                    U: Copy,
                {
                    type Output = #ty<V>;
                    #fn_attrs
                    fn #trait_fn(self, rhs: #ty<U>) -> Self::Output {
                        #ty {
                            #(#fields1)*
                        }
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
                            #(#fields2)*
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
                            #(#fields3)*
                        }
                    }
                }
            },
        ])
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
                self.#f as #to
            }
        });
        let fn_attrs = fn_attrs();
        parse_quote! {
            impl #ty<#from> {
                #fn_attrs
                pub fn #fn_ident(self) -> #ty<#to> {
                    #ty::new(#(#fields1),*)
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
    fn define(self) -> ItemTrait {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        parse_quote! {
            pub trait #trait_ty {
                type Output;
                fn #trait_fn(self) -> Self::Output;
            }
        }
    }

    pub fn impl_for(self, ty: Type, _algebra: Algebra) -> ItemImpl {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let fn_attrs = fn_attrs();

        match self {
            NormOps::Norm2 => {
                parse_quote! {
                    impl<T, U> #trait_ty for #ty<T>
                    where
                        Scalar<U>: GradeProduct<#ty<T>, #ty<T>, Output = Scalar<U>>,
                        #ty<T>: Reverse + Copy,
                        T: std::ops::Mul<Output = U>,
                    {
                        type Output = Scalar<U>;
                        #fn_attrs
                        fn #trait_fn(self) -> Self::Output {
                            Scalar::product(self, self.rev())
                        }
                    }
                }
            }
            NormOps::Norm => {
                parse_quote! {
                    impl<T, U, V> #trait_ty for #ty<T>
                    where
                        Scalar<U>: GradeProduct<#ty<T>, #ty<T>, Output = Scalar<U>>,
                        #ty<T>: Reverse + Copy,
                        T: std::ops::Mul<Output = U>,
                        U: num_sqrt::Sqrt<Output = V>,
                    {
                        type Output = Scalar<V>;
                        #fn_attrs
                        fn #trait_fn(self) -> Self::Output {
                            let scalar = Scalar::product(self, self.rev());
                            num_sqrt::Sqrt::sqrt(scalar)
                        }
                    }
                }
            }
        }
    }

    pub fn trait_ty(self) -> syn::Type {
        match self {
            NormOps::Norm => parse_quote!(Norm),
            NormOps::Norm2 => parse_quote!(Norm2),
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
    pub fn define() -> ItemTrait {
        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();
        parse_quote! {
            pub trait #trait_ty<Rhs> {
                type Output;
                fn #trait_fn(self, rhs: Rhs) -> Self::Output;
            }
        }
    }

    pub fn impl_for(lhs: Type, rhs: Type, algebra: Algebra) -> Option<ItemImpl> {
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
        parse_quote!(Sandwich)
    }

    pub fn trait_fn() -> Ident {
        parse_quote!(sandwich)
    }
}

pub struct Antisandwich;

impl Antisandwich {
    pub fn define() -> ItemTrait {
        let trait_ty = Self::trait_ty();
        let trait_fn = Self::trait_fn();
        parse_quote! {
            pub trait #trait_ty<Rhs> {
                type Output;
                fn #trait_fn(self, rhs: Rhs) -> Self::Output;
            }
        }
    }

    pub fn impl_for(lhs: Type, rhs: Type, algebra: Algebra) -> Option<ItemImpl> {
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
        parse_quote!(Antisandwich)
    }

    pub fn trait_fn() -> Ident {
        parse_quote!(antisandwich)
    }
}

impl InverseOps {
    pub fn define(self) -> ItemTrait {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        parse_quote! {
            pub trait #trait_ty {
                type Output;
                fn #trait_fn(self) -> Self::Output;
            }
        }
    }

    pub fn trait_ty(self) -> syn::Type {
        match self {
            InverseOps::Inverse => parse_quote!(Inverse),
            InverseOps::Unitize => parse_quote!(Unitize),
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
            || (ty.single_blade(algebra) && ty != Type::Grade(0))
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

        Some(match (self, ty) {
            (InverseOps::Inverse, Type::Grade(0)) => {
                parse_quote! {
                    impl<T> #trait_ty for #ty<T>
                    where
                        #ty<T>: Norm2<Output = #ty<T>>,
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
                let norm2 = NormOps::Norm2.trait_ty();
                parse_quote! {
                    impl<T> #trait_ty for #ty<T>
                    where
                        #ty<T>: #norm2<Output = Scalar<T>>
                            + Reverse
                            + std::ops::Mul<Scalar<T>, Output = #ty<T>>
                            + Copy,
                        Scalar<T>: Inverse<Output = Scalar<T>>,
                    {
                        type Output = #ty<T>;
                        #fn_attrs
                        fn #trait_fn(self) -> Self::Output {
                            self.rev() * self.norm2().inv()
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
                let norm = NormOps::Norm.trait_ty();
                parse_quote! {
                    impl<T> #trait_ty for #ty<T>
                    where
                        #ty<T>: #norm<Output = Scalar<T>>
                            + std::ops::Mul<Scalar<T>, Output = #ty<T>>
                            + Copy,
                        Scalar<T>: Inverse<Output = Scalar<T>>,
                    {
                        type Output = Unit<#ty<T>>;
                        #fn_attrs
                        fn #trait_fn(self) -> Self::Output {
                            Unit(self * self.norm().inv())
                        }
                    }
                }
            }
        })
    }

    pub fn tests(self, ty: Type, algebra: Algebra) -> Option<syn::ItemFn> {
        if self.inapplicable(ty, algebra) {
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

        let expr = match self {
            InverseOps::Inverse => quote! {
                let inv = value.inv();
                let product = value.geo(inv);
                let remainder = (product - Scalar { s: 1. }).norm2();
                dbg!(value, inv, product);
                assert!(remainder.s < 1e-10);
            },
            InverseOps::Unitize => quote! {
                let unit = value.unit().value();
                let product = unit.geo(unit.rev());
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
    pub fn define() -> Vec<syn::Item> {
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
            impl<T> #inverse_ty for Unit<T> where T: #reverse_ty {
                type Output = Self;
                #[inline]
                fn #inverse_fn(self) -> Self::Output {
                    Unit(self.0.rev())
                }
            }
        };
        let sandwich_impl = parse_quote! {
            impl<Lhs, Rhs, Int> Sandwich<Rhs> for Unit<Lhs>
            where
                Lhs: Geo<Rhs, Output = Int> + Reverse + Copy,
                Rhs: GradeProduct<Int, Lhs, Output = Rhs>,
            {
                type Output = Rhs;
                #[inline]
                fn sandwich(self, rhs: Rhs) -> Self::Output {
                    let int = self.value().geo(rhs);
                    Rhs::product(int, self.value().rev())
                }
            }
        };
        let antisandwich_impl = parse_quote! {
            impl<Lhs, Rhs, Int> Antisandwich<Rhs> for Unit<Lhs>
            where
                Lhs: Antigeo<Rhs, Output = Int> + Reverse + Copy,
                Rhs: GradeProduct<Int, Lhs, Output = Rhs>,
            {
                type Output = Rhs;
                #[inline]
                fn antisandwich(self, rhs: Rhs) -> Self::Output {
                    let int = self.value().antigeo(rhs);
                    Rhs::product(int, self.value().rev())
                }
            }
        };
        use syn::Item::*;

        IntoIterator::into_iter([
            Struct(item_struct),
            Impl(item_impl),
            Impl(unitize_impl),
            Impl(inverse_impl),
            Impl(sandwich_impl),
            Impl(antisandwich_impl),
        ])
        .chain(ProductOp::iter_all().map(|op| {
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
        .collect()
    }
}

pub struct Sqrt;

impl Sqrt {
    pub fn impl_for(ty: Type, _algebra: Algebra) -> Option<syn::ItemImpl> {
        let sqrt_ty = Self::trait_ty();
        let sqrt_fn = Self::trait_fn();
        match ty {
            Type::Grade(0) => Some(parse_quote! {
                impl<T, U> #sqrt_ty for #ty<T>
                where
                    T: #sqrt_ty<Output = U>,
                {
                    type Output = Scalar<U>;
                    #[inline]
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

        let fn_attrs = fn_attrs();

        Some(parse_quote! {
            impl<T, U, V> std::ops::Div<#rhs<U>> for #lhs<T>
            where
                #lhs<T>: std::ops::Mul<#rhs<U>, Output = #output<V>>,
                #rhs<U>: Inverse<Output = #rhs<U>>,
            {
                type Output = #output<V>;
                #fn_attrs
                fn div(self, rhs: #rhs<U>) -> Self::Output {
                    self * rhs.inv()
                }
            }
        })
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
            }
        }
    }

    #[test]
    fn write_to_file() {
        let path = "../output.rs";
        let output = Algebra::pga3().define().to_string();
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
            #[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
            pub struct Bivector<T> {
                pub xy: T,
                pub xz: T,
                pub yz: T,
            }
        }
        .to_string();
        assert_eq!(expected, tokens);
    }
}

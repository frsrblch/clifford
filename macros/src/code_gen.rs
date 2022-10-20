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
            .chain([Reverse::define()])
            .chain([GradeProduct::define()])
            .chain(NormOps::iter().map(NormOps::define))
            .chain([Sandwich::define()])
            .chain(InverseOps::iter().map(InverseOps::define));

        let structs = self.types().map(move |ty| ty.define(self));

        let impl_grade_fns = self.types().filter_map(|ty| ty.impl_grade_fns(self));

        let impl_zero = self.types().map(|ty| ty.impl_zero(self));

        let impl_product_ops = ProductOp::iter()
            .flat_map(|op| self.types().map(move |ty| (op, ty)))
            .flat_map(|(op, lhs)| self.types().map(move |rhs| (op, lhs, rhs)))
            .filter_map(|(op, lhs, rhs)| op.impl_for(self, lhs, rhs));

        let impl_sum_ops = SumOp::iter()
            .flat_map(|op| self.types().map(move |lhs| (op, lhs)))
            .flat_map(|(op, lhs)| self.types().map(move |rhs| (op, lhs, rhs)))
            .filter_map(|(op, lhs, rhs)| op.impl_for(self, lhs, rhs));

        let impl_neg = self.types().map(|ty| Neg::impl_for(ty, self));

        let impl_rev = self.types().map(|ty| Reverse::impl_for(ty, self));

        let impl_complements = Complement::iter(self)
            .flat_map(|comp| self.types().map(move |ty| ty.impl_complement(self, comp)));

        let scalar_ops =
            ScalarOps::iter().flat_map(|op| self.types().map(move |ty| op.impl_for(ty, self)));

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

        let norm_ops =
            NormOps::iter().flat_map(|op| self.types().map(move |ty| op.impl_for(ty, self)));

        let sandwich_ops = self
            .type_tuples()
            .map(|(lhs, rhs)| Sandwich::impl_for(lhs, rhs, self));

        let unit_items = Unit::define();

        let inverse_ops = InverseOps::iter()
            .flat_map(|op| self.types().filter_map(move |ty| op.impl_for(ty, self)));

        let test_unit_inv =
            InverseOps::iter().flat_map(|op| self.types().filter_map(move |ty| op.tests(ty, self)));

        quote!(
            #(#traits)*
            #(#structs)*
            #(#impl_grade_fns)*
            #(#impl_zero)*
            #(#impl_product_ops)*
            #(#impl_sum_ops)*
            #(#impl_neg)*
            #(#impl_rev)*
            #(#impl_complements)*
            #(#scalar_ops)*
            #(#explicit_scalar_ops)*
            #(#scalar_assign_ops)*
            #(#sum_assign_ops)*
            #(#float_conversion)*
            #(#grade_products)*
            #(#norm_ops)*
            #(#sandwich_ops)*
            #(#inverse_ops)*
            #(#unit_items)*

            #[cfg(test)]
            mod tests {
                use super::*;
                #(#test_unit_inv)*
            }
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

    fn trait_ty() -> syn::Type {
        parse_quote!(Reverse)
    }

    fn trait_fn() -> Ident {
        parse_quote!(rev)
    }
}

pub struct GradeProduct;

impl GradeProduct {
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
        parse_quote!(GradeProduct)
    }

    pub fn trait_fn() -> Ident {
        parse_quote!(product)
    }

    pub fn impl_for(lhs: Type, rhs: Type, out: Type, algebra: Algebra) -> Option<ItemImpl> {
        let blades = lhs
            .iter_blades_unsorted(algebra)
            .flat_map(|lhs| {
                rhs.iter_blades_unsorted(algebra)
                    .map(move |rhs| (lhs, rhs, algebra.mul(lhs, rhs)))
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
                sum = quote! { Default::default() };
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
                    + Default,
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
            let grade_fn = Ident::new(rhs.name_lowercase(), Span::mixed_site());
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

    fn ident(self) -> Ident {
        Ident::new(self.name(), Span::mixed_site())
    }

    fn impl_complement(self, algebra: Algebra, op: Complement) -> ItemImpl {
        let ident = op.ident();
        let fn_ident = op.fn_ident();
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
}

impl Complement {
    fn ident(self) -> Ident {
        match self {
            Self::Dual => parse_quote!(Dual),
            Self::LeftComp => parse_quote!(LeftComp),
            Self::RightComp => parse_quote!(RightComp),
        }
    }

    fn fn_ident(self) -> Ident {
        match self {
            Self::Dual => parse_quote!(dual),
            Self::LeftComp => parse_quote!(left_comp),
            Self::RightComp => parse_quote!(right_comp),
        }
    }

    fn define(self) -> ItemTrait {
        let ident = self.ident();
        let fn_ident = self.fn_ident();
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
    fn field(self, algebga: Algebra) -> Ident {
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
        Self::iter().map(Self::define)
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

    fn trait_ty(self) -> syn::Type {
        match self {
            Self::Geo => parse_quote!(Geo),
            Self::Wedge => parse_quote!(Wedge),
            Self::Dot => parse_quote!(Dot),
        }
    }

    fn trait_fn(self) -> Ident {
        match self {
            Self::Geo => parse_quote!(geo),
            Self::Wedge => parse_quote!(wedge),
            Self::Dot => parse_quote!(dot),
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
                sum.extend(quote!(Default::default()));
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
                        + Default,
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

    fn trait_ty(self) -> syn::Type {
        match self {
            SumOp::Add => parse_quote! { std::ops::Add },
            SumOp::Sub => parse_quote! { std::ops::Sub },
        }
    }

    fn trait_fn(self) -> Ident {
        match self {
            SumOp::Add => parse_quote! { add },
            SumOp::Sub => parse_quote! { sub },
        }
    }
}

impl ScalarOps {
    pub fn impl_for(self, ty: Type, algebra: Algebra) -> ItemImpl {
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        let fn_attrs = fn_attrs();
        let fields = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            quote! { #f: #trait_ty::#trait_fn(self.#f, rhs), }
        });
        parse_quote! {
            impl<T, U, V> #trait_ty<U> for #ty<T>
            where
                T: #trait_ty<U, Output = V>,
                U: Copy,
            {
                type Output = #ty<V>;
                #fn_attrs
                fn #trait_fn(self, rhs: U) -> Self::Output {
                    #ty {
                        #(#fields)*
                    }
                }
            }
        }
    }

    pub fn impl_for_scalar(self, ty: Type, algebra: Algebra) -> Option<[ItemImpl; 2]> {
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

    fn trait_ty(self) -> syn::Type {
        match self {
            Self::AddAssign => parse_quote!(std::ops::AddAssign),
            Self::SubAssign => parse_quote!(std::ops::SubAssign),
        }
    }

    fn trait_fn(self) -> syn::Type {
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
        let fields = ty.iter_blades_unsorted(algebra).map(|blade| {
            let f = blade.field(algebra);
            if matches!(self, Self::ToF32) {
                quote! {
                    #f: {
                        let value = self.#f as #to;
                        debug_assert!(value.is_finite());
                        value
                    },
                }
            } else {
                quote! {
                    #f: self.#f as #to,
                }
            }
        });
        let fn_attrs = fn_attrs();
        parse_quote! {
            impl #ty<#from> {
                #fn_attrs
                pub fn #fn_ident(self) -> #ty<#to> {
                    #ty {
                        #(#fields)*
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

        let where_clause = match self {
            NormOps::Norm2 => quote! {
                where
                    Scalar<T>: GradeProduct<#ty<T>, #ty<T>>,
                    #ty<T>: Reverse + Copy,
            },
            NormOps::Norm => quote! {
                where
                    Scalar<T>: GradeProduct<#ty<T>, #ty<T>>,
                    #ty<T>: Reverse + Copy,
                    T: num_sqrt::Sqrt<Output = T>,
            },
        };

        let expr = match self {
            NormOps::Norm2 => quote! {
                Scalar::product(self, self.rev()).s
            },
            NormOps::Norm => quote! {
                Scalar::product(self, self.rev()).s.sqrt()
            },
        };

        parse_quote! {
            impl<T> #trait_ty for #ty<T> #where_clause {
                type Output = T;
                #fn_attrs
                fn #trait_fn(self) -> Self::Output {
                    #expr
                }
            }
        }
    }

    fn trait_ty(self) -> syn::Type {
        match self {
            NormOps::Norm => parse_quote!(Norm),
            NormOps::Norm2 => parse_quote!(Norm2),
        }
    }

    fn trait_fn(self) -> Ident {
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
                #rhs<W>: #grade_prod_ty<#intermediate<V>, #lhs<T>>,
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

    fn trait_ty() -> syn::Type {
        parse_quote!(Sandwich)
    }

    fn trait_fn() -> Ident {
        parse_quote!(sandwich)
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

    fn inapplicable(self, ty: Type, algebra: Algebra) -> bool {
        algebra.has_negative_bases() || ty == Type::Mv || ty.single_blade(algebra)
    }

    pub fn impl_for(self, ty: Type, algebra: Algebra) -> Option<ItemImpl> {
        if self.inapplicable(ty, algebra) {
            return None;
        }

        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();

        let output = match self {
            InverseOps::Inverse => quote!(#ty<T>),
            InverseOps::Unitize => quote!(Unit<#ty<T>>),
        };

        let fn_attrs = fn_attrs();
        let fn_attrs = quote! {
            #[track_caller]
            #fn_attrs
        };

        let expr = match self {
            InverseOps::Inverse => {
                quote! {
                    let norm2 = self.norm2();
                    if norm2.is_zero() {
                        panic!("divide by zero")
                    }
                    let mult = T::one() / norm2;
                    self.rev() * mult
                }
            }
            InverseOps::Unitize => {
                quote! {
                    let norm = self.norm();
                    if norm.is_zero() {
                        panic!("divide by zero")
                    }
                    let mult = T::one() / norm;
                    Unit(self * mult)
                }
            }
        };

        let type_bounds = match self {
            InverseOps::Unitize => {
                let norm = NormOps::Norm.trait_ty();
                quote! {
                    #ty<T>: #norm<Output = T>,
                }
            }
            InverseOps::Inverse => {
                let norm2 = NormOps::Norm2.trait_ty();
                quote! {
                    #ty<T>: #norm2<Output = T>,
                }
            }
        };

        Some(parse_quote! {
            impl<T> #trait_ty for #ty<T>
            where
                T: num_traits::Zero
                    + num_traits::One
                    + std::ops::Neg<Output = T>
                    + std::ops::Div<Output = T>
                    + Copy,
                #type_bounds
            {
                type Output = #output;
                #fn_attrs
                fn #trait_fn(self) -> Self::Output {
                    #expr
                }
            }
        })
    }

    pub fn tests(self, ty: Type, algebra: Algebra) -> Option<syn::ItemFn> {
        // skip if there's only one blade, and mv which are not easily inverted
        if self.inapplicable(ty, algebra) {
            return None;
        }

        let name = ty.name_lowercase();
        let fn_name = match self {
            InverseOps::Inverse => format!("{}_inv_test", name),
            InverseOps::Unitize => format!("{}_unit_test", name),
        };
        let fn_ident = Ident::new(&fn_name, Span::mixed_site());

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
                assert!(remainder < 1e-10);
            },
            InverseOps::Unitize => quote! {
                let unit = value.unit().value();
                let product = unit.geo(unit.rev());
                let remainder = (product - Scalar { s: 1. }).norm2();
                dbg!(value, unit, product);
                assert!(remainder < 1e-10);
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
        use syn::Item::*;
        vec![
            Struct(item_struct),
            Impl(item_impl),
            Impl(unitize_impl),
            Impl(inverse_impl),
        ]
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
            .define(Algebra::g3())
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

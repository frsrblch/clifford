#![allow(dead_code)]

pub mod blade;
pub mod trait_bounds;

mod binary;
mod parse;
mod unary;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::fmt::Display;

use itertools::Itertools;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote, ToTokens};
use strum::{EnumIter, IntoEnumIterator};

use crate::binary::BinaryTrait;
use crate::blade::Blade;
use crate::trait_bounds::*;
use crate::unary::UnaryTrait;

pub trait IsEven {
    fn is_even(&self) -> bool;
}

impl IsEven for u32 {
    fn is_even(&self) -> bool {
        self & 1 != 1
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Algebra {
    pub bases: Vec<Basis>,
    pub ordering: BladeOrdering,
    pub fields: Vec<Ident>,
}

impl<B> FromIterator<B> for Algebra
where
    Basis: From<B>,
{
    fn from_iter<I: IntoIterator<Item = B>>(iter: I) -> Algebra {
        Algebra::new(iter)
    }
}

mod algebra_new {
    use super::*;

    impl Algebra {
        pub fn new<B, I: IntoIterator<Item = B>>(bases: I) -> Self
        where
            Basis: From<B>,
        {
            Self::new_with_fields(bases, [])
        }

        pub fn new_with_fields<B, I: IntoIterator<Item = B>, F: IntoIterator<Item = Ident>>(
            bases: I,
            field_idents: F,
        ) -> Self
        where
            Basis: From<B>,
        {
            let bases = bases.into_iter().map(Basis::from).collect::<Vec<_>>();

            let dim = bases.len();
            let temp = Algebra {
                bases: bases.clone(),
                fields: vec![],
                ordering: BladeOrdering::new(dim),
            };

            let mut ordering = BladeOrdering::new(dim);

            let mut field_idents: HashMap<Blade, Ident> = {
                let get_blade = |char: char| {
                    bases
                        .iter()
                        .position(|b| b.char == char)
                        .map(|i| Blade(1 << i))
                        .unwrap_or_else(|| panic!("char does not correspond to a basis: {char}"))
                };
                field_idents
                    .into_iter()
                    .map(|field| {
                        let field_str = field.to_string();
                        if field_str == "s" {
                            (Blade(0), field)
                        } else {
                            let blade = field_str
                                .chars()
                                .map(get_blade)
                                .fold(Blade(0), |product, next| temp.geo(product, next));
                            if blade.is_negative() {
                                ordering.flip(blade);
                            }
                            (blade.unsigned(), field)
                        }
                    })
                    .collect()
            };

            let fields = Blades::from(&temp)
                .map(|blade| {
                    if let Some(ident) = field_idents.remove(&blade) {
                        ident
                    } else {
                        blade.field(&temp)
                    }
                })
                .collect();

            Algebra {
                bases,
                fields,
                ordering,
            }
        }
    }

    impl Blade {
        fn field(self, algebra: &Algebra) -> Ident {
            let mut output = String::new();

            let blade_bases = algebra
                .bases
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

            // reverse order of last two bases
            if algebra.ordering.is_flipped(self) {
                let z = output.pop();
                let y = output.pop();
                if let Some(z) = z {
                    output.push(z);
                }
                if let Some(y) = y {
                    output.push(y);
                }
            }

            format_ident!("{output}")
        }
    }
}

impl Algebra {
    pub(crate) fn dim(&self) -> u32 {
        self.bases.len() as u32
    }

    pub(crate) fn has_negative_bases(&self) -> bool {
        self.bases.iter().any(|b| b.square == Square::Neg)
    }

    pub(crate) fn all_bases_positive(&self) -> bool {
        self.bases.iter().all(|b| b.square == Square::Pos)
    }

    pub(crate) fn iter_bases(&self, set: Blade) -> impl Iterator<Item = Basis> + '_ {
        self.bases
            .iter()
            .enumerate()
            .filter_map(move |(i, b)| set.contains(i as u32).then_some(*b))
    }

    pub(crate) fn product<T, U, F: Fn(&Algebra, Blade, Blade) -> Blade>(
        &self,
        lhs: T,
        rhs: U,
        f: F,
    ) -> Option<Type>
    where
        Type: From<T> + From<U>,
        T: Copy,
        U: Copy,
    {
        let f = &f;
        TypeBlades::new(self, lhs)
            .flat_map(|lhs| TypeBlades::new(self, rhs).map(move |rhs| f(self, lhs, rhs)))
            .collect()
    }

    pub(crate) fn geo(&self, lhs: Blade, rhs: Blade) -> Blade {
        let overlap = lhs & rhs;
        let mut product = lhs.product(rhs);
        if self.ordering.is_flipped(lhs)
            ^ self.ordering.is_flipped(rhs)
            ^ self.ordering.is_flipped(product)
        {
            product = -product;
        }
        for basis in self.iter_bases(overlap) {
            product = product.product(basis.square.blade());
        }
        product
    }

    pub(crate) fn dot(&self, lhs: Blade, rhs: Blade) -> Blade {
        let output_grade = lhs.grade().abs_diff(rhs.grade());
        let product = self.geo(lhs, rhs);
        product.filter(|p| p.grade() == output_grade)
    }

    pub(crate) fn wedge(&self, lhs: Blade, rhs: Blade) -> Blade {
        let output_grade = lhs.grade() + rhs.grade();
        let product = self.geo(lhs, rhs);
        product.filter(|b| b.grade() == output_grade)
    }

    pub(crate) fn antigeo(&self, lhs: Blade, rhs: Blade) -> Blade {
        self.anti(lhs, rhs, Self::geo)
    }

    pub(crate) fn antidot(&self, lhs: Blade, rhs: Blade) -> Blade {
        self.anti(lhs, rhs, Self::dot)
    }

    pub(crate) fn antiwedge(&self, lhs: Blade, rhs: Blade) -> Blade {
        self.anti(lhs, rhs, Self::wedge)
    }

    fn anti<F: FnOnce(&Self, Blade, Blade) -> Blade>(&self, lhs: Blade, rhs: Blade, f: F) -> Blade {
        let lhs = self.right_comp(lhs);
        let rhs = self.right_comp(rhs);
        let output = f(self, lhs, rhs);
        self.left_comp(output)
    }

    pub(crate) fn left_con(&self, lhs: Blade, rhs: Blade) -> Blade {
        if lhs.grade() <= rhs.grade() {
            self.dot(lhs, rhs)
        } else {
            Blade::zero()
        }
    }

    pub(crate) fn right_con(&self, lhs: Blade, rhs: Blade) -> Blade {
        if lhs.grade() >= rhs.grade() {
            self.dot(lhs, rhs)
        } else {
            Blade::zero()
        }
    }

    pub(crate) fn right_comp(&self, blade: Blade) -> Blade {
        if blade.is_zero() {
            return Blade::zero();
        }

        let i = Blade::pseudoscalar(self.dim());
        let comp = i ^ blade;
        comp ^ self.geo(blade, comp).sign()
    }

    pub(crate) fn left_comp(&self, blade: Blade) -> Blade {
        if blade.is_zero() {
            return Blade::zero();
        }

        let i = Blade::pseudoscalar(self.dim());
        let comp = i ^ blade;
        comp ^ self.geo(comp, blade).sign()
    }

    /// P × Q = 1/2 (PQ - QP)
    pub(crate) fn commutator(&self, lhs: Blade, rhs: Blade) -> Blade {
        let lr_product = self.geo(lhs, rhs);
        let rl_product = self.geo(rhs, lhs);
        if lr_product == rl_product {
            Blade::zero()
        } else if lr_product == -rl_product {
            lr_product
        } else {
            unreachable!("commutator product")
        }
    }

    pub(crate) fn symmetric_complements(&self) -> bool {
        self.dim() & 1 == 1
    }

    pub(crate) fn grade_blades(&self, grade: u32) -> impl Iterator<Item = Blade> + '_ {
        TypeBlades {
            blades: Blades::from(self),
            ty: Type::Grade(grade),
        }
    }

    pub(crate) fn grades(&self) -> impl Iterator<Item = Type> + '_ {
        self.grade_range().map(Type::Grade)
    }

    pub(crate) fn grade_range(&self) -> std::ops::RangeInclusive<u32> {
        0..=self.dim()
    }

    pub(crate) fn type_blades(&self, ty: Type) -> TypeBlades {
        TypeBlades {
            blades: Blades::from(self),
            ty,
        }
    }

    pub(crate) fn types(&self) -> impl Iterator<Item = Type> + '_ {
        let contains_multiple_grades = move |ty: &Type| {
            let mut single_grade = None;
            for blade in TypeBlades::new(self, *ty) {
                if let Some(g) = single_grade {
                    if blade.grade() != g {
                        return true;
                    }
                } else {
                    single_grade = Some(blade.grade());
                }
            }
            false
        };
        let multiple_grade_types = IntoIterator::into_iter([Type::Motor, Type::Flector, Type::Mv])
            .filter(contains_multiple_grades);
        self.grades().chain(multiple_grade_types)
    }

    pub fn define(&self) -> TokenStream {
        let mut tokens = quote! {
            pub use clifford::*;
            pub use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Neg};
            pub use bytemuck::{Pod, Zeroable};
        };

        for ty in OverType::iter(self) {
            ty.define(self).to_tokens(&mut tokens);
        }

        for i in UnaryTrait::iter()
            .flat_map(|op| OverType::iter(self).map(move |ty| op.define(ty, self)))
        {
            i.to_tokens(&mut tokens);
        }

        for i in BinaryTrait::iter().flat_map(|op| {
            OverType::iter_tuples(self).map(move |(lhs, rhs)| op.define(lhs, rhs, self))
        }) {
            i.to_tokens(&mut tokens);
        }

        let unary_tests = self
            .types()
            .cartesian_product(UnaryTrait::iter())
            .filter_map(|(ty, op)| op.define_tests(ty, self));

        let binary_tests = OverType::iter_tuples(self)
            .cartesian_product(BinaryTrait::iter())
            .filter_map(|((lhs, rhs), op)| op.define_tests(lhs, rhs, self));

        let mut tests = unary_tests.chain(binary_tests).peekable();

        if tests.peek().is_some() {
            let test_mod = quote! {
                #[cfg(test)]
                mod build_tests {
                    use super::*;
                    #(#tests)*
                }
            };
            test_mod.to_tokens(&mut tokens);
        }

        tokens
    }

    fn blade_tuples<T>(&self, lhs: T, rhs: T) -> impl Iterator<Item = (Blade, Blade)> + '_
    where
        Type: From<T>,
    {
        let lhs = TypeBlades::new(self, lhs);
        let rhs = TypeBlades::new(self, rhs);
        lhs.flat_map(move |lhs| rhs.clone().map(move |rhs| (lhs, rhs)))
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Basis {
    pub char: char,
    pub square: Square,
}

impl Basis {
    pub fn pos(char: char) -> Self {
        Self {
            char,
            square: Square::Pos,
        }
    }
    pub fn neg(char: char) -> Self {
        Self {
            char,
            square: Square::Neg,
        }
    }
    pub fn zero(char: char) -> Self {
        Self {
            char,
            square: Square::Zero,
        }
    }
}

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub enum Square {
    #[default]
    Pos,
    Neg,
    Zero,
}

impl Square {
    pub fn basis(self, char: char) -> Basis {
        Basis { char, square: self }
    }

    pub fn blade(self) -> Blade {
        match self {
            Self::Pos => Blade::scalar(),
            Self::Neg => -Blade::scalar(),
            Self::Zero => Blade::zero(),
        }
    }
}

#[derive(Clone)]
pub struct Blades {
    range: std::ops::RangeInclusive<u32>,
}

impl From<u32> for Blades {
    fn from(dim: u32) -> Self {
        let pseudoscalar = Blade::pseudoscalar(dim);
        Blades {
            range: 0..=pseudoscalar.0,
        }
    }
}

impl<'a> From<&'a Algebra> for Blades {
    fn from(algebra: &'a Algebra) -> Self {
        Blades::from(algebra.dim())
    }
}

impl Iterator for Blades {
    type Item = Blade;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(Blade)
    }
}

#[derive(Clone)]
pub struct TypeGrades {
    grades: std::ops::RangeInclusive<u32>,
    ty: Type,
}

impl TypeGrades {
    pub fn new(algebra: &Algebra, ty: Type) -> Self {
        TypeGrades {
            grades: algebra.grade_range(),
            ty,
        }
    }
}

impl Iterator for TypeGrades {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let grade = self.grades.next()?;
            if self.ty.contains_grade(grade) {
                return Some(grade);
            }
        }
    }
}

#[derive(Clone)]
pub struct TypeFields<'a> {
    blades: TypeBlades,
    fields: &'a Vec<Ident>,
}

impl<'a> TypeFields<'a> {
    pub fn new<T>(algebra: &'a Algebra, ty: T) -> Self
    where
        Type: From<T>,
    {
        TypeFields {
            blades: TypeBlades::new(algebra, ty),
            fields: &algebra.fields,
        }
    }
}

impl<'a> Iterator for TypeFields<'a> {
    type Item = (Blade, &'a Ident);

    fn next(&mut self) -> Option<Self::Item> {
        let blade = self.blades.next()?;
        let ident = &self.fields[blade.unsigned().0 as usize];
        Some((blade, ident))
    }
}

#[derive(Clone)]
pub struct TypeBlades {
    blades: Blades,
    ty: Type,
}

impl TypeBlades {
    pub fn new<T>(algebra: &Algebra, ty: T) -> Self
    where
        Type: From<T>,
    {
        TypeBlades {
            blades: Blades::from(algebra),
            ty: Type::from(ty),
        }
    }
}

impl Iterator for TypeBlades {
    type Item = Blade;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let blade = self.blades.next()?;
            if self.ty.contains(blade) {
                return Some(blade);
            }
        }
    }
}

#[derive(Clone)]
pub struct SortedTypeBlades {
    current_grade: Option<u32>,
    current_blades: TypeBlades,
    blades: TypeBlades,
    grades: TypeGrades,
}

impl SortedTypeBlades {
    pub fn new(algebra: &Algebra, ty: Type) -> Self {
        let blades = TypeBlades::new(algebra, ty);
        Self {
            current_grade: None,
            current_blades: blades.clone(),
            blades,
            grades: TypeGrades::new(algebra, ty),
        }
    }
}

impl Iterator for SortedTypeBlades {
    type Item = Blade;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(current_grade) = self.current_grade {
                if let Some(next_blade) = self.current_blades.next() {
                    if next_blade.grade() == current_grade {
                        return Some(next_blade);
                    }
                } else {
                    self.current_grade = Some(self.grades.next()?);
                    self.current_blades = self.blades.clone();
                }
            } else {
                self.current_grade = Some(self.grades.next()?);
            }
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BladeOrdering(Vec<bool>);

impl BladeOrdering {
    fn new(dim: usize) -> Self {
        Self(vec![false; 2usize.pow(dim as u32)])
    }

    fn configure(algebra: &mut Algebra) {
        let dim = algebra.dim();
        for grade in 1..=dim / 2 {
            for blade in Blades::from(dim).filter(|b| b.grade() == grade) {
                let comp = algebra.right_comp(blade);
                if comp.is_negative() && comp.grade() > 1 {
                    algebra.ordering.flip(comp);
                }
            }
        }
    }

    fn flip(&mut self, blade: Blade) {
        if blade.is_zero() {
            return;
        }
        let bool = &mut self.0[blade.unsigned().0 as usize];
        *bool = !*bool;
    }

    fn is_flipped(&self, blade: Blade) -> bool {
        if blade.is_zero() {
            return false;
        }
        self.0[blade.unsigned().0 as usize]
    }

    /// Returns (total, positive)
    #[cfg(test)]
    fn count_positive(algebra: &Algebra, grade: u32) -> (usize, usize) {
        let total = algebra.grade_blades(grade).count();
        let positive = Blades::from(algebra.dim())
            .filter(|b| b.grade() == grade)
            .filter(|b| algebra.right_comp(*b).is_positive())
            .count();
        (total, positive)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Type {
    Grade(u32),
    Motor,
    Flector,
    Mv,
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.ident().fmt(f)
    }
}

impl ToTokens for Type {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ident().to_tokens(tokens);
    }
}

impl From<OverType> for Type {
    fn from(value: OverType) -> Self {
        match value {
            OverType::Type(ty) => ty,
            OverType::Float(_) => Type::Grade(0),
        }
    }
}

impl Type {
    pub fn contains(&self, blade: Blade) -> bool {
        if blade.is_zero() {
            return false;
        }
        self.contains_grade(blade.grade())
    }

    pub fn contains_grade(&self, grade: u32) -> bool {
        match self {
            Type::Mv => true,
            Type::Flector => grade & 1 == 1,
            Type::Motor => grade & 1 != 1,
            Type::Grade(g) => grade == *g,
        }
    }

    pub fn define(self, algebra: &Algebra) -> TokenStream {
        let derive = if TypeBlades::new(algebra, self).nth(1).is_some() {
            quote!(#[derive(Eq, Hash)])
        } else {
            quote!(#[derive(Eq, Ord, PartialOrd, Hash)])
        };
        let ident = self.ident();
        let fields = SortedTypeBlades::new(algebra, self).map(|blade| {
            let field = &algebra.fields[blade];
            quote! {
                pub #field: T,
            }
        });
        let new_params = SortedTypeBlades::new(algebra, self)
            .map(|blade| &algebra.fields[blade])
            .map(|field| quote!(#field: T));
        let new_fields = TypeFields::new(algebra, self).map(|(_, field)| field);
        let grade_fns = TypeGrades::new(algebra, self).filter_map(|grade| {
            let grade = Type::Grade(grade);
            if self == grade {
                return None;
            }
            let ident = grade.fn_ident();
            let fields =
                TypeFields::new(algebra, grade).map(|(_, field)| quote!(#field: self.#field));

            let grade_t = grade.with_type_param(FloatParam::T, MagParam::Mag(Mag::Any));
            Some(quote! {
                #[inline]
                pub fn #ident(self) -> #grade_t {
                    #grade {
                        #(#fields,)*
                        marker: std::marker::PhantomData,
                    }
                }
            })
        });

        let impl_debug = {
            let blades = SortedTypeBlades::new(algebra, self);
            let check_zero = blades
                .clone()
                .map(|blade| {
                    let field = &algebra.fields[blade];
                    quote! {
                        clifford::Zero::is_zero(&self.#field)
                    }
                })
                .collect::<syn::punctuated::Punctuated<_, syn::Token![&&]>>();
            let debug_fields = blades.map(|blade| {
                let field = &algebra.fields[blade];
                let field_lit = syn::LitStr::new(&field.to_string(), field.span());
                quote! {
                    if !clifford::Zero::is_zero(&self.#field) {
                        debug_struct.field(#field_lit, &self.#field);
                    } else {
                        non_exhaustive = true;
                    }
                }
            });
            quote! {
                impl<T: std::fmt::Debug + clifford::Zero, M> std::fmt::Debug for #ident<T, M> {
                    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                        let ty = std::any::type_name::<Self>();

                        let all_zero = #check_zero;
                        if all_zero {
                            return f.debug_tuple(ty).field(&<T as clifford::Zero>::zero()).finish();
                        }

                        let mut debug_struct = f.debug_struct(ty);
                        let mut non_exhaustive = false;
                        #(#debug_fields)*
                        if non_exhaustive {
                            debug_struct.finish_non_exhaustive()
                        } else {
                            debug_struct.finish()
                        }
                    }
                }
            }
        };

        let impl_default_any = {
            let ty_t = quote!(#ident<T, Any>);
            let fields = TypeFields::new(algebra, self).map(|(_, field)| {
                quote! { #field: Default::default() }
            });
            quote! {
                impl<T: Default> Default for #ty_t {
                    fn default() -> Self {
                        #ident {
                            #(#fields,)*
                            marker: std::marker::PhantomData
                        }
                    }
                }
            }
        };
        let impl_default_unit = if self == Type::Motor {
            let ty_t = quote!(#ident<T, Unit>);
            let fields = TypeFields::new(algebra, self).map(|(blade, field)| {
                if blade == Blade::scalar() {
                    quote! { #field: one() }
                } else {
                    quote! { #field: Default::default() }
                }
            });
            quote! {
                impl<T: Default + One> Default for #ty_t {
                    fn default() -> Self {
                        #ident {
                            #(#fields,)*
                            marker: std::marker::PhantomData
                        }
                    }
                }
            }
        } else {
            quote!()
        };

        let impl_copy_clone = {
            let ty_t = quote!(#ident<T, A>);
            let fields = TypeFields::new(algebra, self).map(|(_, field)| {
                quote! { #field: self.#field.clone() }
            });
            quote! {
                impl<T: Clone, A> Clone for #ty_t {
                    fn clone(&self) -> Self {
                        #ident {
                            #(#fields,)*
                            marker: std::marker::PhantomData,
                        }
                    }
                }
                impl<T: Copy, A> Copy for #ty_t {}
            }
        };

        let map_fields =
            TypeFields::new(algebra, self).map(|(_, field)| quote!(#field: f(self.#field)));

        let numeric_traits = if self == Type::Grade(0) {
            let numeric_traits = impl_numeric_traits_for_scalar();
            let number = impl_number_for_scalar(algebra);
            quote! {
                #numeric_traits
                #number
            }
        } else {
            quote!()
        };

        let const_basis = {
            let consts = TypeFields::new(algebra, self).map(|(bf, f)| {
                let m = if algebra.dot(bf, bf).is_zero() {
                    Mag::Any
                } else {
                    Mag::Unit
                };
                let const_ident = Ident::new(&f.to_string().to_uppercase(), f.span());
                let fields = TypeFields::new(algebra, self).map(|(bg, g)| {
                    if bf == bg {
                        quote!(#g: <T as clifford::OneConst>::ONE)
                    } else {
                        quote!(#g: <T as clifford::ZeroConst>::ZERO)
                    }
                });
                quote! {
                    pub const #const_ident: #ident<T, #m> = #ident {
                        #(#fields,)*
                        marker: std::marker::PhantomData
                    };
                }
            });
            quote! {
                impl<T> #ident<T> where T: clifford::OneConst + clifford::ZeroConst {
                    #(#consts)*
                }
            }
        };

        let allow_clippy_too_many_arguments = TypeBlades::new(algebra, self)
            .nth(7)
            .is_some()
            .then(|| quote!(#[allow(clippy::too_many_arguments)]));

        let assert_fields =
            TypeFields::new(algebra, self).map(|(_, field)| quote!(#field: self.#field));

        let d = format!(
            "Represents a {} in a {}-dimensional space",
            self.fn_ident(),
            algebra.dim()
        );

        let n = proc_macro2::Literal::usize_unsuffixed(TypeBlades::new(algebra, self).count());
        let array_fields = &TypeFields::new(algebra, self)
            .map(|(_, f)| f)
            .collect::<Vec<_>>();

        quote! {
            #[doc = #d]
            #[repr(C)]
            #derive
            pub struct #ident<T = f64, M = Any> {
                #(#fields)*
                pub marker: std::marker::PhantomData<M>,
            }

            #impl_debug
            #impl_default_any
            #impl_default_unit
            #impl_copy_clone

            impl<T> #ident<T, Any> {
                #[inline]
                #allow_clippy_too_many_arguments
                pub const fn new(#(#new_params),*) -> #ident<T, Any> {
                    #ident {
                        #(#new_fields,)*
                        marker: std::marker::PhantomData,
                    }
                }
            }
            impl<M> #ident<f32, M> {
                #[inline]
                pub fn to_f64(self) -> #ident<f64, M> {
                    self.map(|t| t as f64).assert()
                }
            }
            impl<M> #ident<f64, M> {
                #[inline]
                pub fn to_f32(self) -> #ident<f32, M> {
                    self.map(|t| t as f32).assert()
                }
            }
            impl<T, M> #ident<T, M> {
                #[inline]
                pub fn map<F: Fn(T) -> U, U>(self, f: F) -> #ident<U> {
                    #ident {
                        #(#map_fields,)*
                        marker: std::marker::PhantomData,
                    }
                }
                #(#grade_fns)*
                #[doc = "Assert that given value is of the given magnitude state N"]
                #[inline]
                pub fn assert<N>(self) -> #ident<T, N> {
                    #ident {
                        #(#assert_fields,)*
                        marker: std::marker::PhantomData,
                    }
                }

                #[doc = "Convert the value to an array"]
                #[inline]
                pub fn to_array(self) -> [T; #n] {
                    let #ident { #(#array_fields,)* .. } = self;
                    [#(#array_fields),*]
                }

                #[doc = "Create a value from an array"]
                #[inline]
                pub fn from_array(array: [T; #n]) -> Self {
                    let [#(#array_fields),*] = array;
                    #ident {
                        #(#array_fields,)*
                        marker: std::marker::PhantomData
                    }
                }
            }

            #const_basis

            #numeric_traits
        }
    }

    pub fn ident(&self) -> Ident {
        format_ident!("{}", self.as_str())
    }

    pub fn fn_ident(&self) -> Ident {
        let lowercase = self.as_str().to_lowercase();
        format_ident!("{lowercase}")
    }

    fn as_str(&self) -> &'static str {
        match *self {
            Type::Grade(grade) => match grade {
                0 => "Scalar",
                1 => "Vector",
                2 => "Bivector",
                3 => "Trivector",
                4 => "Quadvector",
                5 => "Pentavector",
                6 => "Hexavector",
                7 => "Heptavector",
                8 => "Octovector",
                9 => "Nonavector",
                10 => "Decavector",
                _ => unimplemented!("grade out of range: {grade}"),
            },
            Type::Motor => "Motor",
            Type::Flector => "Flector",
            Type::Mv => "Multivector",
        }
    }

    pub fn complement(&self, algebra: &Algebra) -> Self {
        let dim = algebra.bases.len();
        let odd = dim & 1 == 1;
        match self {
            Type::Grade(grade) => Type::Grade(algebra.bases.len() as u32 - grade),
            Type::Motor => {
                if odd {
                    if dim > 2 {
                        Type::Flector
                    } else {
                        Type::Grade(1)
                    }
                } else {
                    Type::Motor
                }
            }
            Type::Flector => {
                if odd {
                    Type::Motor
                } else {
                    Type::Flector
                }
            }
            Type::Mv => Type::Mv,
        }
    }

    pub fn with_type_param<F, M>(self, ty: F, mag: M) -> ParameterizedType
    where
        FloatParam: From<F>,
        MagParam: From<M>,
    {
        ParameterizedType::new(OverType::Type(self), ty.into(), mag.into())
    }
}

impl std::ops::Add for Type {
    type Output = Type;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (lhs, rhs) if lhs == Type::Mv || rhs == Type::Mv => Type::Mv,
            (lhs, rhs) if (lhs == rhs) => lhs,
            (Type::Grade(lhs), Type::Grade(rhs)) => match (lhs.is_even(), rhs.is_even()) {
                (true, true) => Type::Motor,
                (false, false) => Type::Flector,
                _ => Type::Mv,
            },
            (Type::Grade(g), Type::Motor) | (Type::Motor, Type::Grade(g)) => {
                if g.is_even() {
                    Type::Motor
                } else {
                    Type::Mv
                }
            }
            (Type::Flector, Type::Grade(g)) | (Type::Grade(g), Type::Flector) => {
                if g.is_even() {
                    Type::Mv
                } else {
                    Type::Flector
                }
            }
            (Type::Flector, Type::Motor) | (Type::Motor, Type::Flector) => Type::Mv,
            _ => unreachable!(),
        }
    }
}

impl FromIterator<Blade> for Option<Type> {
    fn from_iter<T: IntoIterator<Item = Blade>>(iter: T) -> Self {
        let mut has_even = false;
        let mut has_odd = false;
        enum Grade {
            Single(u32),
            Multiple,
        }
        let mut grade = None;
        for blade in iter {
            if blade.is_zero() {
                continue;
            }
            if blade.grade().is_even() {
                has_even = true;
            } else {
                has_odd = true;
            }
            grade = match grade {
                None => Some(Grade::Single(blade.grade())),
                Some(Grade::Single(g)) => {
                    if g == blade.grade() {
                        Some(Grade::Single(g))
                    } else {
                        Some(Grade::Multiple)
                    }
                }
                Some(Grade::Multiple) => Some(Grade::Multiple),
            };
        }
        if let Some(Grade::Single(grade)) = grade {
            Some(Type::Grade(grade))
        } else {
            match (has_even, has_odd) {
                (true, true) => Some(Type::Mv),
                (true, false) => Some(Type::Motor),
                (false, true) => Some(Type::Flector),
                (false, false) => None,
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum OverType {
    Type(Type),
    Float(Float),
}

impl From<Type> for OverType {
    fn from(value: Type) -> Self {
        OverType::Type(value)
    }
}

impl From<Float> for OverType {
    fn from(value: Float) -> Self {
        OverType::Float(value)
    }
}

impl OverType {
    fn access_field(&self, var: &TokenStream, blade: Blade, algebra: &Algebra) -> TokenStream {
        match self {
            OverType::Type(_) => {
                let field = &algebra.fields[blade];
                quote!(#var.#field)
            }
            OverType::Float(_) => quote!(#var),
        }
    }

    pub fn contains_blade(&self, blade: Blade) -> bool {
        match self {
            OverType::Type(ty) => ty.contains(blade),
            OverType::Float(_) => blade.grade() == 0,
        }
    }

    pub fn contains(&self, rhs: Self) -> bool {
        match (Type::from(*self), Type::from(rhs)) {
            (lhs, Type::Grade(g)) => lhs.contains_grade(g),
            (Type::Grade(_), Type::Motor) => false,
            (Type::Grade(_), Type::Flector) => false,
            (Type::Grade(_), Type::Mv) => false,
            (Type::Motor, Type::Motor) => true,
            (Type::Motor, Type::Flector) => false,
            (Type::Motor, Type::Mv) => false,
            (Type::Flector, Type::Motor) => false,
            (Type::Flector, Type::Flector) => true,
            (Type::Flector, Type::Mv) => false,
            (Type::Mv, Type::Motor) => true,
            (Type::Mv, Type::Flector) => true,
            (Type::Mv, Type::Mv) => true,
        }
    }

    pub fn iter(algebra: &Algebra) -> impl Iterator<Item = Self> + '_ {
        let types = algebra.types().map(OverType::Type);
        let floats = Float::iter().map(OverType::Float);
        types.chain(floats)
    }

    pub fn iter_tuples(algebra: &Algebra) -> impl Iterator<Item = (Self, Self)> + '_ {
        Self::iter(algebra).flat_map(|lhs| Self::iter(algebra).map(move |rhs| (lhs, rhs)))
    }

    pub fn define(&self, algebra: &Algebra) -> Impl<TokenStream> {
        match self {
            OverType::Type(ty) => Impl::Actual(ty.define(algebra)),
            OverType::Float(_) => Impl::External,
        }
    }

    #[track_caller]
    pub fn with_type_param<T, M>(self, t: T, m: M) -> ParameterizedType
    where
        FloatParam: From<T>,
        MagParam: From<M>,
    {
        ParameterizedType::new(self, t.into(), m.into())
    }

    pub fn is_float(self) -> bool {
        matches!(self, OverType::Float(_))
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, EnumIter, Hash)]
pub enum Float {
    F32,
    F64,
}

impl ToTokens for Float {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Float::F32 => quote!(f32).to_tokens(tokens),
            Float::F64 => quote!(f64).to_tokens(tokens),
        }
    }
}

pub enum Impl<T> {
    /// Unimplemented
    None,
    /// Implemented outside this crate
    External,
    /// Implemented elsewhere by this crate
    Internal,
    /// The actual implementation
    Actual(T),
}

impl<T: ToTokens> ToTokens for Impl<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if let Impl::Actual(t) = self {
            t.to_tokens(tokens);
        }
    }
}

impl<T> Impl<T> {
    pub fn map<F: Fn(T) -> U, U>(self, f: F) -> Impl<U> {
        match self {
            Impl::None => Impl::None,
            Impl::External => Impl::External,
            Impl::Internal => Impl::Internal,
            Impl::Actual(t) => Impl::Actual(f(t)),
        }
    }
}

#[derive(Debug)]
pub struct Value {
    ty: Type,
    blades: Vec<f64>,
}

impl Value {
    pub fn new(algebra: &Algebra) -> Self {
        let blades = vec![0f64; 2usize.pow(algebra.dim())];
        Value {
            ty: Type::Grade(0),
            blades,
        }
    }

    pub fn gen(ty: Type, algebra: &Algebra) -> Self {
        let mut rng = rand::thread_rng();
        let mut blades = vec![0f64; 2usize.pow(algebra.dim())];
        for blade in TypeBlades::new(algebra, ty) {
            blades[blade] = rand::Rng::gen_range(&mut rng, -1.0..1.0);
        }
        Value { ty, blades }
    }

    pub fn unit(&mut self, algebra: &Algebra) {
        let norm = self.norm(algebra);
        self.blades.iter_mut().for_each(|b| *b /= norm);
    }

    pub fn try_gen_unit(ty: Type, algebra: &Algebra) -> Option<Self> {
        for _ in 0..20 {
            let mut value = Self::gen(ty, algebra);
            value.unit(algebra);
            if value.is_finite() {
                return Some(value);
            }
        }
        None
    }

    pub fn is_finite(&self) -> bool {
        self.blades.iter().all(|f| f.is_finite())
    }

    pub fn norm(&self, algebra: &Algebra) -> f64 {
        Blades::from(algebra)
            .zip(&self.blades)
            .filter_map(|(blade, value)| {
                if !algebra.dot(blade, blade).is_zero() {
                    Some(value * value)
                } else {
                    None
                }
            })
            .sum::<f64>()
            .sqrt()
    }

    pub fn mul(&self, rhs: &Self, algebra: &Algebra) -> Option<Value> {
        let mut output = Value::new(algebra);
        output.ty = algebra.product(self.ty, rhs.ty, Algebra::geo)?;
        for (l, r) in algebra.blade_tuples(self.ty, rhs.ty) {
            let o = algebra.geo(l, r);
            if o.is_zero() {
                continue;
            } else if o.is_positive() {
                output.blades[o] += self.blades[l] * rhs.blades[r];
            } else {
                output.blades[o] -= self.blades[l] * rhs.blades[r];
            }
        }

        Some(output)
    }

    pub fn is_unit(&self, algebra: &Algebra) -> bool {
        (self.norm(algebra) - 1.).abs() < 1e-10
    }
}

fn impl_numeric_traits_for_scalar() -> TokenStream {
    quote! {

        macro_rules! impl_float_const_for_scalar {
            ($($fn_:ident,)*) => {
                impl<T> FloatConst for Scalar<T> where T: FloatConst {
                    $(
                        #[allow(non_snake_case)]
                        fn $fn_() -> Self {
                            Scalar::new(T::$fn_())
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

        impl<T> std::ops::Rem for Scalar<T>
        where
            T: std::ops::Rem<Output = T>,
        {
            type Output = Scalar<T>;
            #[inline]
            fn rem(self, rhs: Self) -> Self::Output {
                Scalar {
                    s: std::ops::Rem::rem(self.s, rhs.s),
                    marker: std::marker::PhantomData,
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

fn impl_number_for_scalar(algebra: &Algebra) -> TokenStream {
    if UnaryTrait::Inverse.no_impl(OverType::Type(Type::Grade(0)), algebra) {
        return quote!();
    }

    quote! {
        impl<T: Sqrt> Sqrt for Scalar<T> {
            type Output = Scalar<T::Output>;
            #[inline]
            fn sqrt(self) -> Self::Output {
                self.map(Sqrt::sqrt)
            }
        }

        impl<T: Ln> Ln for Scalar<T> {
            type Output = Scalar<T::Output>;
            #[inline]
            fn ln(self) -> Self::Output {
                self.map(Ln::ln)
            }
        }

        impl<T: Exp> Exp for Scalar<T> {
            type Output = Scalar<T::Output>;
            #[inline]
            fn exp(self) -> Self::Output {
                self.map(Exp::exp)
            }
        }

        impl<T: Trig> Trig for Scalar<T> {
            type Output = Scalar<T::Output>;

            const TAU: Self = Scalar {
                s: T::TAU,
                marker: std::marker::PhantomData,
            };
            #[inline]
            fn sin(self) -> Self::Output {
                self.map(Trig::sin)
            }
            #[inline]
            fn cos(self) -> Self::Output {
                self.map(Trig::cos)
            }
            #[inline]
            fn sin_cos(self) -> (Self::Output, Self::Output) {
                let (sin, cos) = self.s.sin_cos();
                (Scalar::new(sin), Scalar::new(cos))
            }
            #[inline]
            fn tan(self) -> Self::Output {
                self.map(Trig::tan)
            }
            #[inline]
            fn sinh(self) -> Self::Output {
                self.map(Trig::sinh)
            }
            #[inline]
            fn cosh(self) -> Self::Output {
                self.map(Trig::cosh)
            }
            #[inline]
            fn tanh(self) -> Self::Output {
                self.map(Trig::tanh)
            }
        }

        impl<T: InvTrig> InvTrig for Scalar<T> {
            type Output = Scalar<T::Output>;

            #[inline]
            fn asin(self) -> Self::Output {
                self.map(InvTrig::asin)
            }
            #[inline]
            fn acos(self) -> Self::Output {
                self.map(InvTrig::acos)
            }
            #[inline]
            fn atan(self) -> Self::Output {
                self.map(InvTrig::atan)
            }
            #[inline]
            fn atan2(self, y: Self) -> Self::Output {
                Scalar::new(self.s.atan2(y.s))
            }
            #[inline]
            fn asinh(self) -> Self::Output {
                self.map(InvTrig::asinh)
            }
            #[inline]
            fn acosh(self) -> Self::Output {
                self.map(InvTrig::acosh)
            }
            #[inline]
            fn atanh(self) -> Self::Output {
                self.map(InvTrig::atanh)
            }
        }

        impl<T: FromF64> FromF64 for Scalar<T> {
            #[inline]
            fn from_f64(value: f64) -> Self {
                Scalar::new(T::from_f64(value))
            }
        }
    }
}

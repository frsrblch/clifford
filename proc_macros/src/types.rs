use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use syn::punctuated::Punctuated;
use syn::token::Comma;

pub struct Zero;

impl Zero {
    pub fn ty() -> syn::Type {
        syn::parse_str("crate::Zero").unwrap()
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Algebra {
    one: u8,
    neg_one: u8,
    zero: u8,
}

impl Algebra {
    pub fn new(one: u8, neg_one: u8, zero: u8) -> Self {
        assert!(
            one + zero + neg_one <= 6,
            "too many bases to define algebra"
        );

        Self { one, zero, neg_one }
    }

    pub fn basis(&self, index: u8) -> Basis {
        Basis(index, *self)
    }

    pub fn square(&self, index: u8) -> Product {
        if index == 0 {
            panic!("zero is not a valid index");
        }

        if index <= self.one {
            Product::Pos(self.scalar())
        } else if index <= self.one + self.zero {
            Product::Zero
        } else if index <= self.one + self.zero + self.neg_one {
            Product::Neg(self.scalar())
        } else {
            panic!("index out of range: {index}");
        }
    }

    pub fn types(&self) -> impl Iterator<Item = Type> + '_ {
        Type::iter(self)
    }

    pub fn bases(&self) -> impl Iterator<Item = Basis> + '_ {
        (1..=self.dimensions()).into_iter().map(|i| self.basis(i))
    }

    pub fn grades(&self) -> impl Iterator<Item = Grade> + '_ {
        (0..=self.dimensions()).into_iter().map(|i| self.grade(i))
    }

    pub fn pseudoscalar(&self) -> Blade {
        let mut set = BladeSet(0);
        for i in 1..=self.dimensions() {
            set.insert(i);
        }
        self.blade(set)
    }

    pub fn scalar(&self) -> Blade {
        self.blade(BladeSet::default())
    }

    pub fn grade(&self, grade: u8) -> Grade {
        match grade {
            grade if grade <= self.dimensions() => Grade(grade, *self),
            _ => panic!("invalid grade: {grade}"),
        }
    }

    pub fn blades(&self) -> impl Iterator<Item = Blade> + '_ {
        self.grades().flat_map(|grade| {
            self.blades_unsorted()
                .filter(move |blade| blade.grade() == grade)
        })
    }

    pub fn subalgebras(&self) -> impl Iterator<Item = SubAlgebra> + '_ {
        [SubAlgebra::Even(*self), SubAlgebra::Odd(*self)].into_iter()
    }

    fn blades_unsorted(&self) -> impl Iterator<Item = Blade> + '_ {
        (0..=self.pseudoscalar().0 .0)
            .into_iter()
            .map(|set| self.blade(BladeSet(set)))
    }

    fn to_blades_unsorted(self) -> impl Iterator<Item = Blade> + 'static {
        (0..=self.pseudoscalar().0 .0)
            .into_iter()
            .map(move |set| self.blade(BladeSet(set)))
    }

    pub fn blade<B: Into<BladeSet>>(&self, set: B) -> Blade {
        Blade(set.into(), *self)
    }

    pub fn dimensions(&self) -> u8 {
        self.one + self.zero + self.neg_one
    }

    pub fn is_homogenous(&self) -> bool {
        match (self.one, self.neg_one, self.zero) {
            (3, 0, 1) | (2, 0, 1) => true,
            _ => false,
        }
    }

    pub fn null_basis(&self) -> Option<Blade> {
        if self.is_homogenous() {
            Some(self.blade(1 << self.one))
        } else {
            None
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Product {
    Pos(Blade),
    Zero,
    Neg(Blade),
}

impl Product {
    pub fn with_blade(&self, blade: Blade) -> Self {
        match self {
            Product::Pos(_) => Product::Pos(blade),
            Product::Neg(_) => Product::Neg(blade),
            Product::Zero => Product::Zero,
        }
    }

    #[allow(dead_code)]
    pub fn is_pos(&self) -> bool {
        matches!(self, Product::Pos(_))
    }

    pub fn is_neg(&self) -> bool {
        matches!(self, Product::Neg(_))
    }

    pub fn blade(self) -> Option<Blade> {
        match self {
            Self::Pos(blade) | Self::Neg(blade) => Some(blade),
            Self::Zero => None,
        }
    }
}

impl std::ops::Mul for Product {
    type Output = Product;

    fn mul(self, rhs: Self) -> Self::Output {
        use Product::*;
        match (self, rhs) {
            (Zero, _) | (_, Zero) => Zero,
            (Pos(lhs), Pos(rhs)) | (Neg(lhs), Neg(rhs)) => lhs * rhs,
            (Pos(lhs), Neg(rhs)) | (Neg(lhs), Pos(rhs)) => -(lhs * rhs),
        }
    }
}

impl std::ops::Neg for Product {
    type Output = Self;
    fn neg(self) -> Self::Output {
        match self {
            Self::Zero => Self::Zero,
            Self::Pos(b) => Self::Neg(b),
            Self::Neg(b) => Self::Pos(b),
        }
    }
}

impl std::ops::MulAssign for Product {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Basis(pub u8, pub Algebra);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Grade(pub u8, pub Algebra);

impl ToBlades for Grade {
    fn blades(&self) -> Box<dyn Iterator<Item = Blade> + '_> {
        Box::new(self.blades())
    }
}

impl Grade {
    pub fn blades(&self) -> impl Iterator<Item = Blade> + '_ {
        self.1.blades_unsorted().filter(|b| b.0.len() == self.0)
    }
    pub fn to_blades(self) -> impl Iterator<Item = Blade> + 'static {
        self.1
            .to_blades_unsorted()
            .filter(move |b| b.0.len() == self.0)
    }

    pub fn is_scalar(&self) -> bool {
        self.0 == 0
    }

    pub fn is_even(self) -> bool {
        self.0 % 2 == 0
    }

    pub fn is_odd(self) -> bool {
        self.0 % 2 == 1
    }

    pub fn is_pseudoscalar(&self) -> bool {
        self.blades().next() == Some(self.1.pseudoscalar())
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Blade(pub BladeSet, pub Algebra);

impl Blade {
    pub fn contains(&self, basis: Basis) -> bool {
        self.1 == basis.1 && self.0.contains(basis.0)
    }

    pub fn grade(&self) -> Grade {
        Grade(self.0.len(), self.1)
    }

    pub fn is_even(self) -> bool {
        self.grade().is_even()
    }

    pub fn is_odd(self) -> bool {
        self.grade().is_odd()
    }

    pub fn dot(self, rhs: Self) -> Product {
        let multiplier = self * rhs;

        if let Some(blade) = multiplier.blade() {
            let a = self.0.len();
            let b = rhs.0.len();
            let max = a.max(b);
            let min = a.min(b);
            let grade = max - min;

            if blade.0.len() == grade {
                multiplier
            } else {
                Product::Zero
            }
        } else {
            Product::Zero
        }
    }

    pub fn wedge(self, rhs: Self) -> Product {
        let multiplier = self * rhs;

        if let Some(blade) = multiplier.blade() {
            let a = self.0.len();
            let b = rhs.0.len();
            let grade = a + b;

            if blade.0.len() == grade {
                multiplier
            } else {
                Product::Zero
            }
        } else {
            Product::Zero
        }
    }
}

impl std::ops::Mul for Blade {
    type Output = Product;

    fn mul(mut self, mut rhs: Self) -> Self::Output {
        let mut multiplier = Product::Pos(self.1.scalar());

        for base in self.1.bases() {
            if rhs.contains(base) {
                let swaps = ((base.0 + 1)..=self.1.dimensions())
                    .into_iter()
                    .filter(|&b| self.0.contains(b))
                    .count()
                    + (1..base.0)
                        .into_iter()
                        .filter(|&b| rhs.0.contains(b))
                        .count();

                if swaps % 2 == 1 {
                    multiplier *= Product::Neg(self.1.scalar());
                }

                if self.0.contains(base.0) {
                    multiplier *= self.1.square(base.0);
                }
                self.0.flip(base.0);
                rhs.0.flip(base.0);
            }
        }

        debug_assert!(rhs.0.is_empty());

        multiplier.with_blade(self)
    }
}

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct BladeSet(pub u64);

impl From<u64> for BladeSet {
    fn from(value: u64) -> Self {
        BladeSet(value)
    }
}

impl BladeSet {
    pub fn contains(&self, index: u8) -> bool {
        let flag = Self::flag(index);
        self.0 & flag == flag
    }

    pub fn insert(&mut self, index: u8) {
        self.0 |= Self::flag(index);
    }

    pub fn flip(&mut self, index: u8) {
        let flag = Self::flag(index);
        self.0 ^= flag;
    }

    fn flag(index: u8) -> u64 {
        assert!(index > 0, "index cannot be zero (e1 is stored at index 0)");
        1 << (index - 1)
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub fn len(&self) -> u8 {
        self.0.count_ones() as u8
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SubAlgebra {
    Even(Algebra),
    Odd(Algebra),
}

impl SubAlgebra {
    pub fn blades(&self) -> impl Iterator<Item = Blade> + '_ {
        let algebra = self.algebra();
        let by_grade = match self {
            SubAlgebra::Even(_) => |b: &Blade| b.is_even(),
            SubAlgebra::Odd(_) => |b: &Blade| b.is_odd(),
        };
        algebra.blades().filter(by_grade)
    }

    pub fn algebra(&self) -> &Algebra {
        match self {
            SubAlgebra::Even(a) => a,
            SubAlgebra::Odd(a) => a,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Type {
    Zero(Algebra),
    Grade(Grade),
    SubAlgebra(SubAlgebra),
}

impl Type {
    pub fn algebra(&self) -> Algebra {
        match self {
            Self::Zero(a) => *a,
            Self::Grade(g) => g.1,
            Self::SubAlgebra(s) => *s.algebra(),
        }
    }

    pub fn iter(alg: &Algebra) -> impl Iterator<Item = Type> + '_ {
        std::iter::once(Type::Zero(*alg))
            .chain(alg.grades().map(Type::Grade))
            .chain(alg.subalgebras().map(Type::SubAlgebra))
    }

    pub fn blades(&self) -> Box<dyn Iterator<Item = Blade> + '_> {
        match self {
            Type::Zero(_) => Box::new(std::iter::empty()),
            Type::Grade(grade) => Box::new(grade.blades()),
            Type::SubAlgebra(sub) => Box::new(sub.blades()),
        }
    }

    pub fn from_iter<T: IntoIterator<Item = Blade>>(iter: T, alg: Algebra) -> Self {
        let mut all_even = true;
        let mut all_odd = true;
        let mut grades = std::collections::HashSet::new();

        for blade in iter {
            let grade = blade.0.len();

            grades.insert(blade.1.grade(grade));

            if grade % 2 == 0 {
                all_odd = false;
            } else {
                all_even = false;
            }
        }

        if grades.is_empty() {
            return Type::Zero(alg);
        }

        if grades.len() == 1 {
            return Type::Grade(grades.into_iter().next().unwrap());
        }

        if all_even {
            return Type::SubAlgebra(SubAlgebra::Even(grades.into_iter().next().unwrap().1));
        }

        if all_odd {
            return Type::SubAlgebra(SubAlgebra::Odd(grades.into_iter().next().unwrap().1));
        }

        unimplemented!("multi vector")
    }

    pub fn is_zero(&self) -> bool {
        matches!(self, Type::Zero(_))
    }

    pub fn is_scalar(&self) -> bool {
        match self {
            Self::Grade(grade) => grade.is_scalar(),
            _ => false,
        }
    }

    pub fn is_local(&self) -> bool {
        match self {
            Type::Zero(_) => false,
            Type::Grade(g) => !g.is_scalar(),
            _ => true,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TypeMv {
    Zero(Algebra),
    Grade(Grade),
    Multivector(Multivector),
}

impl ToBlades for TypeMv {
    fn blades(&self) -> Box<dyn Iterator<Item = Blade> + '_> {
        match self {
            Self::Zero(_) => Box::new(std::iter::empty()),
            Self::Grade(g) => Box::new(g.blades()),
            Self::Multivector(mv) => mv.blades(),
        }
    }
}

#[derive(Default)]
pub struct WhereClause {
    pub bounds: Punctuated<Bound, Comma>,
}

impl ToTokens for WhereClause {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let bounds = &self.bounds;
        if !bounds.is_empty() {
            tokens.extend(quote! {
                where #bounds
            })
        }
    }
}

impl FromIterator<Bound> for WhereClause {
    fn from_iter<T: IntoIterator<Item = Bound>>(iter: T) -> Self {
        Self {
            bounds: iter.into_iter().collect(),
        }
    }
}

pub struct Bound {
    pub ty: Box<dyn ToTokens>,
    pub trait_ty: TokenStream,
    pub generics: Generics,
}

impl ToTokens for Bound {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let Bound {
            ty,
            trait_ty,
            generics,
        } = self;
        tokens.extend(quote! {
            #ty: #trait_ty #generics
        });
    }
}

#[derive(Clone)]
pub enum Generic {
    Generic(Ident),
    Concrete(syn::Type),
    Output(Ident),
}

impl Generic {
    pub fn ident(&self) -> Option<Ident> {
        match self {
            Self::Generic(i) | Self::Output(i) => Some(i.clone()),
            _ => None,
        }
    }
}

// TODO create enum TypeParam { Generic(Ident), TokenStream } with Ord
#[derive(Default)]
pub struct Generics {
    pub params: Punctuated<Generic, Comma>,
}

impl FromIterator<Generic> for Generics {
    fn from_iter<T: IntoIterator<Item = Generic>>(iter: T) -> Self {
        Generics {
            params: iter.into_iter().collect(),
        }
    }
}

impl ToTokens for Generic {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Generic(i) => i.to_tokens(tokens),
            Self::Concrete(ty) => ty.to_tokens(tokens),
            Self::Output(ty) => tokens.extend(quote! { Output = #ty }),
        }
    }
}

impl ToTokens for Generics {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if !self.params.is_empty() {
            let params = &self.params;
            tokens.extend(quote! { <#params> })
        }
    }
}

impl Generics {
    pub fn iter(&self) -> syn::punctuated::Iter<Generic> {
        self.params.iter()
    }
}

impl std::ops::Index<usize> for Generics {
    type Output = Generic;
    fn index(&self, index: usize) -> &Self::Output {
        self.params.index(index)
    }
}

impl TypeMv {
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Grade(g) if g.is_scalar())
    }

    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Zero(_))
    }

    pub fn generics(&self, suffix: &str) -> Generics {
        match self {
            Self::Multivector(mv) => Generics {
                params: mv
                    .1
                    .grades()
                    .map(|g| Generic::Generic(g.generic(suffix)))
                    .collect(),
            },
            _ => Generics::default(),
        }
    }

    pub fn iter(algebra: &Algebra) -> impl Iterator<Item = Self> + '_ {
        use std::iter::once;
        once(TypeMv::Zero(*algebra))
            .chain(algebra.grades().map(Self::Grade))
            .chain(once(TypeMv::Multivector(Multivector(
                GradeSet::default(),
                *algebra,
            ))))
    }

    pub fn from_iter<I: IntoIterator<Item = Product>>(iter: I, algebra: Algebra) -> Self {
        let mut grades = std::collections::HashSet::<Grade>::default();
        for product in iter {
            if let Some(blade) = product.blade() {
                grades.insert(blade.grade());
            }
        }
        match grades.len() {
            0 => Self::Zero(algebra),
            1 => Self::Grade(grades.into_iter().next().unwrap()),
            _ => {
                let mut mv = Multivector::new(algebra);

                for grade in grades.into_iter() {
                    mv.0.insert(grade);
                }

                Self::Multivector(mv)
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Multivector(pub GradeSet, pub Algebra);

impl Multivector {
    pub fn new(algebra: Algebra) -> Self {
        Self(GradeSet::default(), algebra)
    }

    pub fn generics(&self, suffix: &str) -> Generics {
        if self.0 == GradeSet::default() {
            Generics {
                params: self
                    .1
                    .grades()
                    .map(|g| Generic::Generic(g.generic(suffix)))
                    .collect(),
            }
        } else {
            Generics {
                params: self
                    .1
                    .grades()
                    .map(|g| {
                        if self.0.contains(g) {
                            Generic::Concrete(g.type_ident())
                        } else {
                            Generic::Concrete(Zero::ty())
                        }
                    })
                    .collect(),
            }
        }
    }

    pub fn blades(&self) -> Box<dyn Iterator<Item = Blade> + '_> {
        if self.0 == GradeSet::default() {
            Box::new(self.1.grades().flat_map(|g| g.to_blades()))
        } else {
            Box::new(
                self.1
                    .grades()
                    .filter(|g| self.0.contains(*g))
                    .flat_map(|g| g.to_blades()),
            )
        }
    }

    pub fn grades(&self) -> impl Iterator<Item = Grade> + '_ {
        self.1.grades().filter(|g| self.0.contains(*g))
    }
}

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct GradeSet(u64);

impl GradeSet {
    pub fn contains(&self, grade: Grade) -> bool {
        if *self == GradeSet::default() {
            return true;
        }

        let flag = GradeSet::flag(grade);
        self.0 & flag == flag
    }

    pub fn insert(&mut self, grade: Grade) {
        self.0 |= GradeSet::flag(grade);
    }

    fn flag(grade: Grade) -> u64 {
        1 << grade.0
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ProductOp {
    Mul,
    Geometric,
    Dot,
    Wedge,
}

pub trait ToBlades {
    fn blades(&self) -> Box<dyn Iterator<Item = Blade> + '_>;
}

impl ProductOp {
    pub fn output_contains(&self, lhs: impl ToBlades, rhs: impl ToBlades, output: Grade) -> bool {
        lhs.blades()
            .flat_map(|lhs| {
                rhs.blades()
                    .filter_map(move |rhs| self.call(lhs, rhs).blade())
            })
            .any(|g| g.grade() == output)
    }

    pub fn iter() -> impl Iterator<Item = Self> + 'static {
        [Self::Mul, Self::Geometric, Self::Dot, Self::Wedge].into_iter()
    }

    pub fn call(&self, lhs: Blade, rhs: Blade) -> Product {
        match self {
            ProductOp::Mul => lhs * rhs,
            ProductOp::Geometric => lhs * rhs,
            ProductOp::Dot => lhs.dot(rhs),
            ProductOp::Wedge => lhs.wedge(rhs),
        }
    }

    pub fn trait_ty(&self) -> TokenStream {
        match self {
            ProductOp::Mul => quote! { std::ops::Mul },
            ProductOp::Geometric => quote! { crate::Geometric },
            ProductOp::Dot => quote! { crate::Dot },
            ProductOp::Wedge => quote! { crate::Wedge },
        }
    }

    pub fn trait_fn(&self) -> TokenStream {
        match self {
            ProductOp::Mul => quote! { mul },
            ProductOp::Geometric => quote! { geo },
            ProductOp::Dot => quote! { dot },
            ProductOp::Wedge => quote! { wedge },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn no_bases() {
        Algebra::new(0, 0, 0).square(0);
    }

    #[test]
    #[should_panic]
    fn too_many_bases() {
        Algebra::new(7, 0, 0);
    }

    #[test]
    fn one_d() {
        let alg = Algebra::new(1, 0, 0);
        assert!(alg.square(1).is_pos());
        assert!(std::panic::catch_unwind(|| alg.square(2)).is_err());
    }

    #[test]
    fn pga() {
        let alg = Algebra::new(3, 0, 1);
        assert!(alg.square(3).is_pos());
        assert_eq!(Product::Zero, alg.square(4));
        assert!(std::panic::catch_unwind(|| alg.square(5)).is_err());
    }

    #[test]
    fn cga() {
        let bases = Algebra::new(4, 1, 0);
        assert!(bases.square(4).is_pos());
        assert!(bases.square(5).is_neg());
        assert!(std::panic::catch_unwind(|| bases.square(6)).is_err());
    }

    #[test]
    fn blade_set_contains() {
        assert!(BladeSet(0).is_empty());
        assert!(!BladeSet(0).contains(1));
        assert!(BladeSet(1).contains(1));
        assert_eq!(1, BladeSet(0b_0001).len());
        assert_eq!(1, BladeSet(0b_0010).len());
        assert_eq!(2, BladeSet(0b_0011).len());
        assert_eq!(8, BladeSet(0b_1111_1111).len());
    }

    #[test]
    fn blade_set_insert() {
        let mut set = BladeSet(0);
        set.insert(1);
        set.insert(3);

        assert_eq!(2, set.len());
        assert_eq!(BladeSet(0b_0101), set);
    }

    #[test]
    fn bases_pseudovector() {
        let alg = Algebra::new(3, 0, 0);

        assert_eq!(alg.blade(0b_0111), alg.pseudoscalar());
    }

    #[test]
    fn grade_blades() {
        let alg = Algebra::new(3, 1, 0);

        assert_eq!(1, alg.grade(0).blades().count());
        assert_eq!(4, alg.grade(1).blades().count());
        assert_eq!(6, alg.grade(2).blades().count());
        assert_eq!(4, alg.grade(3).blades().count());
        assert_eq!(1, alg.grade(4).blades().count());
    }

    #[test]
    fn blade_multiplication() {
        let alg = Algebra::new(3, 1, 1);
        let e12 = alg.blade(0b_0011);
        let e23 = alg.blade(0b_0110);
        let e24 = alg.blade(0b_1010);
        let e5 = alg.blade(0b_1_0000);

        assert_eq!(Product::Neg(alg.blade(0)), e12 * e12);
        assert_eq!(Product::Pos(alg.blade(0b_0101)), e12 * e23);
        assert_eq!(Product::Pos(alg.blade(0b_1001)), e12 * e24);
        assert_eq!(Product::Zero, e24 * e24);
        assert_eq!(Product::Neg(alg.blade(0)), e5 * e5);
    }

    #[test]
    fn blade_dot() {
        let alg = Algebra::new(3, 1, 0);
        let e12 = alg.blade(0b_0011);
        let e23 = alg.blade(0b_0110);
        let e34 = alg.blade(0b_1100);

        assert_eq!(Product::Neg(alg.blade(0)), e12.dot(e12));
        assert_eq!(Product::Zero, e12.dot(e23));
        assert_eq!(Product::Zero, e12.dot(e34));
    }

    #[test]
    fn blade_wedge() {
        let alg = Algebra::new(3, 1, 0);
        let e12 = alg.blade(0b_0011);
        let e23 = alg.blade(0b_0110);
        let e34 = alg.blade(0b_1100);

        assert_eq!(Product::Zero, e12.wedge(e12));
        assert_eq!(Product::Zero, e12.wedge(e23));
        assert_eq!(Product::Pos(alg.blade(0b_1111)), e12.wedge(e34));
    }
}

// #[derive(Debug, Copy, Clone, Eq, PartialEq)]
// pub enum SumOp {
//     Add,
//     Sub,
// }

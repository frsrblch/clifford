use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use syn::parse_quote;

pub struct Zero;

impl Zero {
    pub fn ident() -> Ident {
        parse_quote! { Zero }
    }

    pub fn ty() -> syn::Type {
        let ident = Self::ident();
        parse_quote! { #ident }
    }

    pub fn expr() -> syn::Expr {
        let ident = Self::ident();
        parse_quote! { #ident }
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

    pub fn basis(self, index: u8) -> Basis {
        Basis(index, self)
    }

    pub fn square(self, index: u8) -> Product {
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

    pub fn types(self) -> impl Iterator<Item = Type> {
        Type::iter(self)
    }

    pub fn bases(self) -> impl Iterator<Item = Basis> {
        (1..=self.dimensions())
            .into_iter()
            .map(move |i| self.basis(i))
    }

    pub fn grades(self) -> Grades {
        Grades::new(self)
    }

    pub fn pseudoscalar(self) -> Blade {
        let mut set = BladeSet(0);
        for i in 1..=self.dimensions() {
            set.insert(i);
        }
        self.blade(set)
    }

    pub fn scalar(self) -> Blade {
        self.blade(BladeSet::default())
    }

    pub fn grade(self, grade: u8) -> Grade {
        match grade {
            grade if grade <= self.dimensions() => Grade(grade, self),
            _ => panic!("invalid grade: {grade}"),
        }
    }

    pub fn blades(self) -> Blades {
        // self.grades().flat_map(move |grade| {
        //     self.blades_unsorted()
        //         .filter(move |blade| blade.grade() == grade)
        // })
        Blades::new(self)
    }

    pub fn subalgebras(self) -> impl Iterator<Item = SubAlgebra> {
        [SubAlgebra::Even(self), SubAlgebra::Odd(self)].into_iter()
    }

    fn blades_unsorted(self) -> BladesUnsorted {
        BladesUnsorted::new(self)
    }

    pub fn blade<B: Into<BladeSet>>(self, set: B) -> Blade {
        Blade(set.into(), self)
    }

    pub fn dimensions(self) -> u8 {
        self.one + self.zero + self.neg_one
    }

    pub fn is_homogenous(self) -> bool {
        match (self.one, self.neg_one, self.zero) {
            (3, 0, 1) | (2, 0, 1) => true,
            _ => false,
        }
    }

    pub fn null_basis(self) -> Option<Blade> {
        if self.is_homogenous() {
            Some(self.blade(1 << self.one))
        } else {
            None
        }
    }

    pub fn mv(self) -> Multivector {
        Multivector::new(self)
    }
}

#[derive(Clone)]
pub struct Blades {
    grades: Grades,
    blades: Option<GradeBlades>,
}

impl Blades {
    pub fn new(algebra: Algebra) -> Self {
        let grades = algebra.grades();
        Self {
            grades: grades,
            blades: None,
        }
    }
}

impl Iterator for Blades {
    type Item = Blade;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(blades) = &mut self.blades {
                if let Some(blade) = blades.next() {
                    return Some(blade);
                }
            }
            let grade = self.grades.next()?;
            self.blades = Some(GradeBlades::new(grade));
        }
    }
}

#[derive(Clone)]
pub struct GradeBlades {
    grade: Grade,
    blades_unsorted: BladesUnsorted,
}

impl GradeBlades {
    pub fn new(grade: Grade) -> Self {
        GradeBlades {
            grade,
            blades_unsorted: grade.1.blades_unsorted(),
        }
    }
}

impl Iterator for GradeBlades {
    type Item = Blade;
    fn next(&mut self) -> Option<Self::Item> {
        for blade in self.blades_unsorted.by_ref() {
            if blade.grade() == self.grade {
                return Some(blade);
            }
        }
        None
    }
}

#[derive(Clone)]
pub struct Grades {
    range: std::ops::RangeInclusive<u8>,
    algebra: Algebra,
}

impl Grades {
    pub fn new(algebra: Algebra) -> Self {
        Grades {
            range: 0..=algebra.dimensions(),
            algebra,
        }
    }
}

impl Iterator for Grades {
    type Item = Grade;
    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|i| self.algebra.grade(i))
    }
}

#[derive(Clone)]
pub struct BladesUnsorted {
    range: std::ops::RangeInclusive<u64>,
    algebra: Algebra,
}

impl BladesUnsorted {
    pub fn new(algebra: Algebra) -> Self {
        BladesUnsorted {
            range: 0..=(algebra.pseudoscalar().0 .0),
            algebra,
        }
    }
}

impl Iterator for BladesUnsorted {
    type Item = Blade;
    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|set| self.algebra.blade(set))
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Product {
    Pos(Blade),
    Zero,
    Neg(Blade),
}

impl Product {
    pub fn with_blade(self, blade: Blade) -> Self {
        match self {
            Product::Pos(_) => Product::Pos(blade),
            Product::Neg(_) => Product::Neg(blade),
            Product::Zero => Product::Zero,
        }
    }

    pub fn filter<F: Fn(Blade) -> bool>(self, f: F) -> Self {
        match self {
            Product::Zero => self,
            Product::Pos(b) | Product::Neg(b) => {
                if f(b) {
                    self
                } else {
                    Product::Zero
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn is_pos(self) -> bool {
        matches!(self, Product::Pos(_))
    }

    pub fn is_neg(self) -> bool {
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

impl IntoIterator for Grade {
    type Item = Blade;
    type IntoIter = impl Iterator<Item = Blade>;
    fn into_iter(self) -> Self::IntoIter {
        Grade::blades(self)
    }
}

impl Grade {
    pub fn blades(self) -> GradeBlades {
        GradeBlades::new(self)
    }

    pub fn is_scalar(self) -> bool {
        self.0 == 0
    }

    pub fn is_even(self) -> bool {
        self.0 % 2 == 0
    }

    pub fn is_odd(self) -> bool {
        self.0 % 2 == 1
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Blade(pub BladeSet, pub Algebra);

impl Blade {
    pub fn contains(self, basis: Basis) -> bool {
        self.1 == basis.1 && self.0.contains(basis.0)
    }

    pub fn grade(self) -> Grade {
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
                    .filter(|b| self.0.contains(*b))
                    .count()
                    + (1..base.0)
                        .into_iter()
                        .filter(|b| rhs.0.contains(*b))
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
    pub fn contains(self, index: u8) -> bool {
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

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn len(self) -> u8 {
        self.0.count_ones() as u8
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SubAlgebra {
    Even(Algebra),
    Odd(Algebra),
}

impl SubAlgebra {
    pub fn blades(self) -> impl Iterator<Item = Blade> {
        let algebra = self.algebra();
        let by_grade = match self {
            SubAlgebra::Even(_) => |b: &Blade| b.is_even(),
            SubAlgebra::Odd(_) => |b: &Blade| b.is_odd(),
        };
        algebra.blades().filter(by_grade)
    }

    pub fn algebra(self) -> Algebra {
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
    pub fn algebra(self) -> Algebra {
        match self {
            Self::Zero(a) => a,
            Self::Grade(g) => g.1,
            Self::SubAlgebra(s) => s.algebra(),
        }
    }

    pub fn iter(alg: Algebra) -> impl Iterator<Item = Type> {
        std::iter::once(Type::Zero(alg))
            .chain(alg.grades().map(Type::Grade))
            .chain(alg.subalgebras().map(Type::SubAlgebra))
    }

    pub fn blades(self) -> Box<dyn Iterator<Item = Blade>> {
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

    pub fn is_zero(self) -> bool {
        matches!(self, Type::Zero(_))
    }

    pub fn is_scalar(self) -> bool {
        match self {
            Self::Grade(grade) => grade.is_scalar(),
            _ => false,
        }
    }

    pub fn is_local(self) -> bool {
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

impl IntoIterator for TypeMv {
    type Item = Blade;
    type IntoIter = TypeMvBlades;
    fn into_iter(self) -> Self::IntoIter {
        TypeMv::blades(self)
    }
}

impl TypeMv {
    pub fn contains(&self, product: Product) -> bool {
        if let Some(blade) = product.blade() {
            self.blades().any(|b| b == blade)
        } else {
            false
        }
    }

    pub fn is_generic(&self) -> bool {
        match self {
            TypeMv::Multivector(mv) => mv.is_generic(),
            _ => false,
        }
    }

    pub fn algebra(self) -> Algebra {
        match self {
            Self::Zero(a) => a,
            Self::Grade(g) => g.1,
            Self::Multivector(mv) => mv.1,
        }
    }

    pub fn blades(self) -> TypeMvBlades {
        match self {
            Self::Zero(_) => TypeMvBlades::Zero,
            Self::Grade(g) => TypeMvBlades::Grade(g.blades()),
            Self::Multivector(mv) => TypeMvBlades::Multivector(mv.blades()),
        }
    }

    pub fn grades(self) -> TypeMvGrades {
        match self {
            Self::Zero(_) => TypeMvGrades::Zero,
            Self::Grade(g) => TypeMvGrades::Grade(std::iter::once(g)),
            Self::Multivector(mv) => TypeMvGrades::Multivector(mv.grades()),
        }
    }

    pub fn is_mv(self) -> bool {
        matches!(self, TypeMv::Multivector(_))
    }
}

#[derive(Clone)]
pub enum TypeMvGrades {
    Zero,
    Grade(std::iter::Once<Grade>),
    Multivector(MvGrades),
}

impl Iterator for TypeMvGrades {
    type Item = Grade;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Zero => None,
            Self::Grade(grade) => grade.next(),
            Self::Multivector(grades) => grades.next(),
        }
    }
}

#[derive(Clone)]
pub enum TypeMvBlades {
    Zero,
    Grade(GradeBlades),
    Multivector(MvBlades),
}

impl Iterator for TypeMvBlades {
    type Item = Blade;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Zero => None,
            Self::Grade(blades) => blades.next(),
            Self::Multivector(blades) => blades.next(),
        }
    }
}

impl TypeMv {
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Grade(g) if g.is_scalar())
    }

    pub fn generics(self, suffix: &str) -> Box<dyn Iterator<Item = Ident> + '_> {
        assert!(!suffix.is_empty());
        match self {
            Self::Multivector(mv) => Box::new(mv.1.grades().map(|g| g.generic(suffix))),
            _ => Box::new(std::iter::empty()),
        }
    }

    pub fn iter(algebra: Algebra) -> impl Iterator<Item = Self> {
        use std::iter::once;
        once(TypeMv::Zero(algebra))
            .chain(algebra.grades().map(Self::Grade))
            .chain(once(TypeMv::Multivector(algebra.mv())))
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
                    mv.insert(grade);
                }
                Self::Multivector(mv)
            }
        }
    }
}

pub fn cartesian_product<Lhs, Rhs, F>(
    lhs: Lhs,
    rhs: Rhs,
    op: F,
) -> impl Iterator<Item = (Blade, Blade, Product)>
where
    Lhs: IntoIterator<Item = Blade>,
    Rhs: IntoIterator<Item = Blade> + Clone,
    F: Fn(Blade, Blade) -> Product + Copy,
{
    lhs.into_iter().flat_map(move |lhs| {
        rhs.clone()
            .into_iter()
            .map(move |rhs| (lhs, rhs, op(lhs, rhs)))
    })
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Multivector(pub GradeSet, pub Algebra);

impl Multivector {
    pub fn new(algebra: Algebra) -> Self {
        Self(GradeSet::default(), algebra)
    }

    pub fn is_generic(&self) -> bool {
        self.0 == GradeSet::default()
    }

    pub fn insert(&mut self, grade: Grade) {
        self.0.insert(grade);
    }

    pub fn generic_parameters(&self, suffix: &str) -> Vec<syn::Type> {
        self.type_generics(suffix)
    }

    pub fn type_parameters(&self, suffix: &str) -> Vec<syn::Type> {
        if self.is_generic() {
            self.type_generics(suffix)
        } else {
            self.1
                .grades()
                .map(|g| {
                    if self.0.contains(g) {
                        g.ty()
                    } else {
                        Zero::ty()
                    }
                })
                .collect()
        }
    }

    pub fn type_generics(&self, suffix: &str) -> Vec<syn::Type> {
        self.1
            .grades()
            .map(|g| {
                let ty = g.generic(suffix);
                parse_quote! { #ty }
            })
            .collect()
    }

    pub fn blades(self) -> MvBlades {
        MvBlades::new(self)
    }

    pub fn grades(self) -> MvGrades {
        MvGrades::new(self)
    }
}

impl From<Grade> for Multivector {
    fn from(grade: Grade) -> Self {
        let mut mv = Multivector::new(grade.1);
        mv.insert(grade);
        mv
    }
}

#[derive(Clone)]
pub struct MvBlades {
    grades: MvGrades,
    blades: Option<GradeBlades>,
}

impl MvBlades {
    pub fn new(multivector: Multivector) -> Self {
        Self {
            grades: multivector.grades(),
            blades: None,
        }
    }
}

impl Iterator for MvBlades {
    type Item = Blade;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(blades) = &mut self.blades {
                if let Some(blade) = blades.next() {
                    return Some(blade);
                }
            }
            let grade = self.grades.next()?;
            self.blades = Some(grade.blades());
        }
    }
}

#[derive(Clone)]
pub struct MvGrades {
    set: GradeSet,
    grades: Grades,
}

impl MvGrades {
    pub fn new(multivector: Multivector) -> Self {
        Self {
            set: multivector.0,
            grades: multivector.1.grades(),
        }
    }
}

impl Iterator for MvGrades {
    type Item = Grade;
    fn next(&mut self) -> Option<Self::Item> {
        for grade in self.grades.by_ref() {
            if self.set.contains(grade) {
                return Some(grade);
            }
        }
        None
    }
}

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct GradeSet(u64);

impl GradeSet {
    pub fn contains(&self, grade: Grade) -> bool {
        if *self == Self::default() {
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

impl std::ops::BitOr for GradeSet {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ProductOp {
    Mul,
    Geometric,
    Dot,
    Wedge,
    Grade(Grade),
}

impl ProductOp {
    pub fn is_local(self) -> bool {
        match self {
            Self::Mul => false,
            Self::Geometric | Self::Dot | Self::Wedge | Self::Grade(_) => true,
        }
    }

    pub fn output_contains(
        self,
        lhs: impl IntoIterator<Item = Blade>,
        rhs: impl IntoIterator<Item = Blade> + Clone,
        output: Grade,
    ) -> bool {
        cartesian_product(lhs, rhs, self)
            .filter_map(|(_, _, p)| p.blade())
            .any(|b| b.grade() == output)
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        [Self::Mul, Self::Geometric, Self::Dot, Self::Wedge].into_iter()
    }

    pub fn iter_all(algebra: Algebra) -> impl Iterator<Item = Self> {
        Self::iter().chain(algebra.grades().map(ProductOp::Grade))
    }

    pub fn call(self, lhs: Blade, rhs: Blade) -> Product {
        match self {
            ProductOp::Mul => lhs * rhs,
            ProductOp::Geometric => lhs * rhs,
            ProductOp::Dot => lhs.dot(rhs),
            ProductOp::Wedge => lhs.wedge(rhs),
            ProductOp::Grade(grade) => {
                let product = lhs * rhs;
                product.filter(|b| b.grade() == grade)
            }
        }
    }

    pub fn expr(
        self,
        lhs: TypeMv,
        lhs_blade: Blade,
        rhs: TypeMv,
        rhs_blade: Blade,
        target: Blade,
    ) -> Option<syn::Expr> {
        let product = self.call(lhs_blade, rhs_blade);

        if product.blade() != Some(target) {
            return None;
        }

        if product.is_pos() {
            let lhs = access_field(lhs, lhs_blade, quote!(self));
            let rhs = access_field(rhs, rhs_blade, quote!(rhs));
            Some(parse_quote! { #lhs * #rhs })
        } else {
            let lhs = access_field(lhs, lhs_blade, quote!(self));
            let rhs = access_field(rhs, rhs_blade, quote!(rhs));
            Some(parse_quote! { -(#lhs * #rhs) })
        }
    }

    pub fn ty(&self) -> syn::Type {
        match self {
            ProductOp::Mul => syn::parse_str("std::ops::Mul").unwrap(),
            ProductOp::Geometric => syn::parse_str("crate::Geometric").unwrap(),
            ProductOp::Dot => syn::parse_str("crate::Dot").unwrap(),
            ProductOp::Wedge => syn::parse_str("crate::Wedge").unwrap(),
            ProductOp::Grade(grade) => syn::parse_str(&format!("{}Product", grade.name())).unwrap(),
        }
    }

    pub fn fn_ident(&self) -> Ident {
        let str = match self {
            ProductOp::Mul => "mul",
            ProductOp::Geometric => "geo",
            ProductOp::Dot => "dot",
            ProductOp::Wedge => "wedge",
            ProductOp::Grade(grade) => {
                let name = grade.name().to_lowercase();
                let str = &format!("{name}_prod");
                return Ident::new(str, Span::mixed_site());
            }
        };
        Ident::new(str, Span::mixed_site())
    }

    pub fn output_mv(self, lhs: TypeMv, rhs: TypeMv) -> TypeMv {
        // if matches!(lhs, TypeMv::Multivector(mv) if mv.is_generic())
        //     || matches!(rhs, TypeMv::Multivector(mv) if mv.is_generic())
        // {
        //     return TypeMv::Multivector(Multivector::new(algebra));
        // };

        let products = lhs
            .into_iter()
            .flat_map(|lhs| rhs.into_iter().map(move |rhs| (lhs, rhs)))
            .map(|(l, r)| self.call(l, r));
        TypeMv::from_iter(products, lhs.algebra())
    }
}

pub fn access_field(parent: TypeMv, blade: Blade, ident: TokenStream) -> syn::Expr {
    if parent.is_scalar() {
        parse_quote! {
            #ident
        }
    } else {
        let member = match parent {
            TypeMv::Grade(_) => syn::Member::Named(blade.field()),
            TypeMv::Multivector(_) => blade.grade().mv_field(),
            TypeMv::Zero(_) => unreachable!("no fields to access"),
        };

        parse_quote! {
            #ident.#member
        }
    }
}

impl FnOnce<(Blade, Blade)> for ProductOp {
    type Output = Product;
    extern "rust-call" fn call_once(self, args: (Blade, Blade)) -> Self::Output {
        self.call(args.0, args.1)
    }
}

impl FnMut<(Blade, Blade)> for ProductOp {
    extern "rust-call" fn call_mut(&mut self, args: (Blade, Blade)) -> Self::Output {
        FnOnce::call_once(*self, args)
    }
}

impl Fn<(Blade, Blade)> for ProductOp {
    extern "rust-call" fn call(&self, args: (Blade, Blade)) -> Self::Output {
        FnOnce::call_once(*self, args)
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

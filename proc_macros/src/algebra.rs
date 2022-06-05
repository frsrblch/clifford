use proc_macro2::{Span, TokenStream};
pub use quote::{quote, ToTokens};
use std::fmt::Formatter;
use std::iter::{empty, once};
use syn::{parse_quote, parse_str, Expr, Ident, Type};

// TODO reconfigure to accommodate larger geometries (BladeSet, GradeSet, etc)
// TODO pass in specialized grade types (e.g., IdealPoints (pga), DualPlanes (cga))
// TODO pass in alternate blade symbols?

trait Convert<U> {
    fn convert(&self) -> U;
}

impl<T: ToTokens, U: syn::parse::Parse> Convert<U> for T {
    fn convert(&self) -> U {
        syn::parse2(self.to_token_stream()).unwrap()
    }
}

pub struct Zero;

impl Zero {
    pub fn ident() -> Ident {
        parse_quote! { Zero }
    }

    pub fn ty() -> Type {
        Self::ident().convert()
    }

    pub fn expr() -> Expr {
        Self::ident().convert()
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

    pub fn mv(self) -> AlgebraType {
        AlgebraType::Multivector(Multivector::new(self))
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

    #[allow(dead_code)]
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
    type IntoIter = GradeBlades;
    fn into_iter(self) -> Self::IntoIter {
        Grade::blades(self)
    }
}

impl Grade {
    pub fn name(self) -> &'static str {
        match self.0 {
            0 => "Scalar",
            1 => "Vector",
            2 => "Bivector",
            3 => "Trivector",
            4 => "Quadvector",
            5 => "Pentavector",
            6 => "Hexavector",
            _ => unimplemented!("not implemented for grade: {}", self.0),
        }
    }

    pub fn ident(self) -> Ident {
        parse_str(self.name()).unwrap()
    }

    pub fn ty(self) -> Type {
        parse_str(self.name()).unwrap()
    }

    pub fn blades(self) -> GradeBlades {
        GradeBlades::new(self)
    }

    pub fn generic(self, suffix: &str) -> Ident {
        Ident::new(&format!("G{}{}", self.0, suffix), Span::mixed_site())
    }

    pub fn generic_n(self, n: usize) -> Ident {
        Ident::new(&format!("G{}_{}", self.0, n), Span::mixed_site())
    }

    pub fn mv_field(self) -> syn::Member {
        syn::Member::Unnamed(syn::Index::from(self.0 as usize))
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Blade(pub BladeSet, pub Algebra);

impl Blade {
    pub fn field(self) -> Ident {
        if self.is_scalar() {
            Ident::new("value", Span::mixed_site())
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

    pub fn is_scalar(self) -> bool {
        self.0.is_empty()
    }

    pub fn contains(self, basis: Basis) -> bool {
        self.1 == basis.1 && self.0.contains(basis.0)
    }

    pub fn grade(self) -> Grade {
        Grade(self.0.len(), self.1)
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
                return multiplier;
            }
        }

        Product::Zero
    }

    pub fn wedge(self, rhs: Self) -> Product {
        let multiplier = self * rhs;

        if let Some(blade) = multiplier.blade() {
            let a = self.0.len();
            let b = rhs.0.len();
            let grade = a + b;

            if blade.0.len() == grade {
                return multiplier;
            }
        }

        Product::Zero
    }

    pub fn left_contraction(self, rhs: Self) -> Product {
        let multiplier = self * rhs;

        if let Some(blade) = multiplier.blade() {
            let a = self.0.len();
            let b = rhs.0.len();

            if let Some(grade) = b.checked_sub(a) {
                if blade.0.len() == grade {
                    return multiplier;
                }
            }
        }

        Product::Zero
    }

    pub fn right_contraction(self, rhs: Self) -> Product {
        let multiplier = self * rhs;

        if let Some(blade) = multiplier.blade() {
            let a = self.0.len();
            let b = rhs.0.len();

            if let Some(grade) = a.checked_sub(b) {
                if blade.0.len() == grade {
                    return multiplier;
                }
            }
        }

        Product::Zero
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

        debug_assert!(rhs.is_scalar());

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
pub enum AlgebraType {
    Zero(Algebra),
    Float(Algebra),
    Grade(Grade),
    Multivector(Multivector),
}

impl IntoIterator for AlgebraType {
    type Item = Blade;
    type IntoIter = TypeMvBlades;
    fn into_iter(self) -> Self::IntoIter {
        AlgebraType::blades(self)
    }
}

impl AlgebraType {
    #[allow(dead_code)]
    pub fn is_zero(self) -> bool {
        matches!(self, AlgebraType::Zero(_))
    }

    pub fn contains(self, grade: Grade) -> bool {
        match self {
            AlgebraType::Zero(_) | AlgebraType::Float(_) => false,
            AlgebraType::Grade(g) => g == grade,
            AlgebraType::Multivector(mv) => mv.0.contains(grade),
        }
    }

    pub fn algebra(self) -> Algebra {
        match self {
            Self::Zero(a) | Self::Float(a) => a,
            Self::Grade(g) => g.1,
            Self::Multivector(mv) => mv.1,
        }
    }

    pub fn blades(self) -> TypeMvBlades {
        match self {
            Self::Zero(_) => TypeMvBlades::Zero,
            Self::Float(a) => TypeMvBlades::Grade(a.grade(0).blades()),
            Self::Grade(g) => TypeMvBlades::Grade(g.blades()),
            Self::Multivector(mv) => TypeMvBlades::Multivector(mv.blades()),
        }
    }

    pub fn grades(self) -> TypeMvGrades {
        match self {
            Self::Zero(_) => TypeMvGrades::Zero,
            Self::Float(a) => Self::Grade(a.grade(0)).grades(),
            Self::Grade(g) => TypeMvGrades::Grade(once(g)),
            Self::Multivector(mv) => TypeMvGrades::Multivector(mv.grades()),
        }
    }

    pub fn is_generic(self) -> bool {
        matches!(self, AlgebraType::Multivector(mv) if mv.is_generic())
    }
    pub fn is_float(self) -> bool {
        matches!(self, Self::Float(_))
    }

    pub fn is_mv(self) -> bool {
        matches!(self, Self::Multivector(_))
    }

    pub fn generics(self, suffix: &str) -> Box<dyn Iterator<Item = Ident> + '_> {
        match self {
            Self::Multivector(mv) => Box::new(mv.1.grades().map(|g| g.generic(suffix))),
            _ => Box::new(empty()),
        }
    }

    pub fn iter(algebra: Algebra) -> impl Iterator<Item = Self> {
        once(AlgebraType::Zero(algebra))
            .chain(once(AlgebraType::Float(algebra)))
            .chain(algebra.grades().map(Self::Grade))
            .chain(once(algebra.mv()))
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

pub fn cartesian_product<Lhs, Rhs>(
    lhs: Lhs,
    rhs: Rhs,
    op: ProductOp,
) -> impl Iterator<Item = (Blade, Blade, Product)>
where
    Lhs: IntoIterator<Item = Blade>,
    Rhs: IntoIterator<Item = Blade> + Clone,
{
    lhs.into_iter().flat_map(move |lhs| {
        rhs.clone()
            .into_iter()
            .map(move |rhs| (lhs, rhs, op.call(lhs, rhs)))
    })
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Multivector(pub GradeSet, pub Algebra);

impl Multivector {
    pub fn ident() -> Ident {
        parse_str("Multivector").unwrap()
    }

    pub fn new(algebra: Algebra) -> Self {
        Self(GradeSet::default(), algebra)
    }

    pub fn is_generic(self) -> bool {
        self.0 == GradeSet::default()
    }

    pub fn insert(&mut self, grade: Grade) {
        self.0.insert(grade);
    }

    pub fn type_parameters(self, suffix: &str) -> Vec<Type> {
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

    pub fn type_generics(self, suffix: &str) -> Vec<Type> {
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

#[derive(Default, Copy, Clone, Eq, PartialEq)]
pub struct GradeSet(u64);

impl std::fmt::Debug for GradeSet {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.debug_tuple(std::any::type_name::<Self>())
            .field(&format!("{:b}", self.0))
            .finish()
    }
}

impl GradeSet {
    pub fn contains(self, grade: Grade) -> bool {
        if self == Self::default() {
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
    Div,
    LeftContraction,
    RightContraction,
}

impl ProductOp {
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
        [
            Self::Mul,
            Self::Geometric,
            Self::Dot,
            Self::Wedge,
            Self::LeftContraction,
            Self::RightContraction,
            Self::Div,
        ]
        .into_iter()
    }

    pub fn iter_all(algebra: Algebra) -> impl Iterator<Item = Self> {
        Self::iter().chain(algebra.grades().map(ProductOp::Grade))
    }

    pub fn call(self, lhs: Blade, rhs: Blade) -> Product {
        match self {
            ProductOp::Mul | ProductOp::Div => lhs * rhs,
            ProductOp::Geometric => lhs * rhs,
            ProductOp::Dot => lhs.dot(rhs),
            ProductOp::Wedge => lhs.wedge(rhs),
            ProductOp::Grade(grade) => {
                let product = lhs * rhs;
                product.filter(|b| b.grade() == grade)
            }
            ProductOp::LeftContraction => lhs.left_contraction(rhs),
            ProductOp::RightContraction => lhs.right_contraction(rhs),
        }
    }

    pub fn product_expr(
        self,
        lhs: AlgebraType,
        lhs_blade: Blade,
        rhs: AlgebraType,
        rhs_blade: Blade,
        target: Blade,
    ) -> Option<Expr> {
        let product = self.call(lhs_blade, rhs_blade);

        if product.blade() != Some(target) {
            return None;
        }

        let s = match self {
            Self::Div => quote!(/),
            _ => quote!(*),
        };

        if product.is_pos() {
            let lhs = access_field(lhs, lhs_blade, quote!(self));
            let rhs = access_field(rhs, rhs_blade, quote!(rhs));
            Some(parse_quote! { #lhs #s #rhs })
        } else {
            let lhs = access_field(lhs, lhs_blade, quote!(self));
            let rhs = access_field(rhs, rhs_blade, quote!(rhs));
            Some(parse_quote! { -(#lhs #s #rhs) })
        }
    }

    pub fn trait_ty(self) -> Type {
        match self {
            ProductOp::Mul => parse_quote!(std::ops::Mul),
            ProductOp::Geometric => parse_quote!(crate::Geometric),
            ProductOp::Dot => parse_quote!(crate::Dot),
            ProductOp::Wedge => parse_quote!(crate::Wedge),
            ProductOp::Grade(grade) => parse_str(&format!("{}Product", grade.name())).unwrap(),
            ProductOp::Div => parse_quote!(std::ops::Div),
            ProductOp::LeftContraction => parse_quote!(crate::LeftContraction),
            ProductOp::RightContraction => parse_quote!(crate::RightContraction),
        }
    }

    pub fn trait_fn(self) -> Ident {
        match self {
            ProductOp::Mul => parse_quote!(mul),
            ProductOp::Geometric => parse_quote!(geo),
            ProductOp::Dot => parse_quote!(dot),
            ProductOp::Wedge => parse_quote!(wedge),
            ProductOp::Grade(grade) => {
                let str = &format!("{}_prod", grade.name().to_lowercase());
                Ident::new(str, Span::mixed_site())
            }
            ProductOp::Div => parse_quote!(div),
            ProductOp::LeftContraction => parse_quote!(left_contraction),
            ProductOp::RightContraction => parse_quote!(right_contraction),
        }
    }

    pub fn output(self, lhs: AlgebraType, rhs: AlgebraType) -> AlgebraType {
        let products = lhs
            .into_iter()
            .flat_map(|lhs| rhs.into_iter().map(move |rhs| (lhs, rhs)))
            .map(|(l, r)| self.call(l, r));
        AlgebraType::from_iter(products, lhs.algebra())
    }

    pub fn grade_op(self, output: Grade) -> Self {
        match self {
            ProductOp::LeftContraction => ProductOp::LeftContraction,
            ProductOp::RightContraction => ProductOp::RightContraction,
            ProductOp::Div => ProductOp::Div,
            _ => ProductOp::Grade(output),
        }
    }
}

pub fn access_field(parent: AlgebraType, blade: Blade, ident: TokenStream) -> Expr {
    let member = match parent {
        AlgebraType::Float(_) => return ident.convert(),
        AlgebraType::Grade(_) => syn::Member::Named(blade.field()),
        AlgebraType::Multivector(_) => blade.grade().mv_field(),
        AlgebraType::Zero(_) => unreachable!("no fields to access"),
    };
    parse_quote! {
        #ident.#member
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SumOp {
    Add,
    Sub,
    GradeAdd,
    GradeSub,
}

impl SumOp {
    pub fn iter() -> impl Iterator<Item = Self> {
        [Self::Add, Self::Sub, Self::GradeAdd, Self::GradeSub].into_iter()
    }

    pub fn is_sub(self) -> bool {
        matches!(self, Self::Sub | Self::GradeSub)
    }

    pub fn is_grade_op(self) -> bool {
        matches!(self, Self::GradeAdd | Self::GradeSub)
    }

    pub fn trait_ty(self) -> Type {
        match self {
            Self::Add => parse_quote!(std::ops::Add),
            Self::Sub => parse_quote!(std::ops::Sub),
            Self::GradeAdd => parse_quote!(crate::GradeAdd),
            Self::GradeSub => parse_quote!(crate::GradeSub),
        }
    }

    pub fn trait_ty_grade(self) -> Type {
        match self {
            Self::Add | Self::GradeAdd => parse_quote!(crate::GradeAdd),
            Self::Sub | Self::GradeSub => parse_quote!(crate::GradeSub),
        }
    }

    pub fn trait_ty_std(self) -> Type {
        match self {
            Self::Add | Self::GradeAdd => parse_quote!(std::ops::Add),
            Self::Sub | Self::GradeSub => parse_quote!(std::ops::Sub),
        }
    }

    pub fn trait_fn(self) -> Ident {
        match self {
            Self::Add | Self::GradeAdd => parse_quote!(add),
            Self::Sub | Self::GradeSub => parse_quote!(sub),
        }
    }

    pub fn trait_fn_grade(self) -> Ident {
        match self {
            Self::Add | Self::GradeAdd => parse_quote!(add),
            Self::Sub | Self::GradeSub => parse_quote!(sub),
        }
    }

    pub fn trait_fn_std(self) -> Type {
        match self {
            Self::Add | Self::GradeAdd => parse_quote!(add),
            Self::Sub | Self::GradeSub => parse_quote!(sub),
        }
    }

    pub fn products(self, lhs: AlgebraType, rhs: AlgebraType) -> Box<dyn Iterator<Item = Product>> {
        let assert_not_mv = |lhs: AlgebraType, rhs: AlgebraType| {
            assert!(
                !lhs.is_mv() && !rhs.is_mv(),
                "{:?} is not implemented for Multivectors",
                self
            );
        };
        match self {
            Self::Add => Box::new(
                lhs.blades()
                    .map(Product::Pos)
                    .chain(rhs.blades().map(Product::Pos)),
            ),
            Self::Sub => Box::new(
                lhs.blades()
                    .map(Product::Pos)
                    .chain(rhs.blades().map(Product::Neg)),
            ),
            Self::GradeAdd => {
                assert_not_mv(lhs, rhs);
                Box::new(
                    lhs.blades()
                        .map(Product::Pos)
                        .chain(rhs.blades().map(Product::Pos)),
                )
            }
            Self::GradeSub => {
                assert_not_mv(lhs, rhs);
                Box::new(
                    lhs.blades()
                        .map(Product::Pos)
                        .chain(rhs.blades().map(Product::Neg)),
                )
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum UnaryOp {
    Neg,
    Reverse,
    LeftComplement,
    RightComplement,
    // Bulk,
    // Weight,
}

impl UnaryOp {
    pub fn iter() -> impl Iterator<Item = Self> {
        [
            Self::Neg,
            Self::Reverse,
            Self::LeftComplement,
            Self::RightComplement,
            // Self::Bulk,
            // Self::Weight,
        ]
        .into_iter()
    }

    pub fn trait_ty(self) -> Type {
        match self {
            Self::Neg => parse_quote! { std::ops::Neg },
            Self::Reverse => parse_quote! { crate::Reverse },
            Self::LeftComplement => parse_quote! { crate::LeftComplement },
            Self::RightComplement => parse_quote! { crate::RightComplement },
            // Self::Bulk => parse_quote! { crate::Bulk },
            // Self::Weight => parse_quote! { crate::Weight },
        }
    }

    pub fn trait_fn(self) -> Ident {
        match self {
            Self::Neg => parse_quote! { neg },
            Self::Reverse => parse_quote! { rev },
            Self::LeftComplement => parse_quote! { left_comp },
            Self::RightComplement => parse_quote! { right_comp },
            // Self::Bulk => parse_quote! { bulk },
            // Self::Weight => parse_quote! { weight },
        }
    }

    pub fn products(
        self,
        blades: impl IntoIterator<Item = Blade>,
    ) -> impl Iterator<Item = Product> {
        blades.into_iter().map(move |b| self.call(b))
    }

    pub fn call(self, blade: Blade) -> Product {
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
            } // Self::Bulk => {
              //     let null_blade = blade.1.null_basis().unwrap();
              //     match blade.wedge(null_blade) {
              //         Product::Zero => Product::Zero,
              //         _ => Product::Pos(blade),
              //     }
              // }
              // Self::Weight => {
              //     let null_blade = blade.1.null_basis().unwrap();
              //     match blade.wedge(null_blade) {
              //         Product::Zero => Product::Pos(blade),
              //         _ => Product::Zero,
              //     }
              // }
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

    #[test]
    fn blade_left_contraction() {
        let algebra = Algebra::new(3, 0, 1);
        let scalar = algebra.scalar();
        let e1 = algebra.blade(1);

        assert_eq!(Product::Pos(scalar), scalar.left_contraction(scalar));
        assert_eq!(Product::Zero, e1.left_contraction(scalar));
        assert_eq!(Product::Pos(e1), scalar.left_contraction(e1));
        assert_eq!(Product::Pos(scalar), e1.left_contraction(e1));
    }

    #[test]
    fn blade_right_contraction() {
        let algebra = Algebra::new(3, 0, 1);
        let scalar = algebra.scalar();
        let e1 = algebra.blade(1);

        assert_eq!(Product::Pos(scalar), scalar.right_contraction(scalar));
        assert_eq!(Product::Pos(e1), e1.right_contraction(scalar));
        assert_eq!(Product::Zero, scalar.right_contraction(e1));
        assert_eq!(Product::Pos(scalar), e1.right_contraction(e1));
    }
}

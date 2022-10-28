use std::iter::FromIterator;

#[derive(Default, Copy, Clone, Eq, PartialEq)]
pub struct Algebra {
    pub bases: &'static [Basis],
}

impl Algebra {
    pub fn ga2() -> Self {
        Self {
            bases: &[
                Basis {
                    char: 'x',
                    sqr: Square::Pos,
                },
                Basis {
                    char: 'y',
                    sqr: Square::Pos,
                },
            ],
        }
    }

    pub fn ga3() -> Self {
        Self {
            bases: &[
                Basis {
                    char: 'x',
                    sqr: Square::Pos,
                },
                Basis {
                    char: 'y',
                    sqr: Square::Pos,
                },
                Basis {
                    char: 'z',
                    sqr: Square::Pos,
                },
            ],
        }
    }

    pub fn pga2() -> Self {
        Self {
            bases: &[
                Basis {
                    char: 'x',
                    sqr: Square::Pos,
                },
                Basis {
                    char: 'y',
                    sqr: Square::Pos,
                },
                Basis {
                    char: 'w',
                    sqr: Square::Zero,
                },
            ],
        }
    }

    pub fn pga3() -> Self {
        Self {
            bases: &[
                Basis {
                    char: 'x',
                    sqr: Square::Pos,
                },
                Basis {
                    char: 'y',
                    sqr: Square::Pos,
                },
                Basis {
                    char: 'z',
                    sqr: Square::Pos,
                },
                Basis {
                    char: 'w',
                    sqr: Square::Zero,
                },
            ],
        }
    }

    pub fn cga3() -> Self {
        Self {
            bases: &[
                Basis {
                    char: 'x',
                    sqr: Square::Pos,
                },
                Basis {
                    char: 'y',
                    sqr: Square::Pos,
                },
                Basis {
                    char: 'z',
                    sqr: Square::Pos,
                },
                Basis {
                    char: 'n',
                    sqr: Square::Pos,
                },
                Basis {
                    char: 'm',
                    sqr: Square::Neg,
                },
            ],
        }
    }

    pub fn has_negative_bases(self) -> bool {
        self.bases.iter().any(|b| b.sqr == Square::Neg)
    }

    pub fn iter_bases(self, set: Blade) -> impl Iterator<Item = Basis> {
        self.bases
            .iter()
            .enumerate()
            .filter_map(move |(i, b)| set.contains(i as u32).then_some(*b))
    }

    pub fn geo(self, lhs: Blade, rhs: Blade) -> Blade {
        if Blade::is_zero(lhs | rhs) {
            return Blade::zero();
        }
        let overlap = lhs & rhs;
        let mut product = lhs.product(rhs);
        for basis in self.iter_bases(overlap) {
            product = product.product(basis.sqr());
        }
        product
    }

    pub fn dot(self, lhs: Blade, rhs: Blade) -> Blade {
        let max = lhs.grade().max(rhs.grade());
        let min = lhs.grade().min(rhs.grade());
        let product = self.geo(lhs, rhs);
        if product.grade() == max - min {
            product
        } else {
            Blade::zero()
        }
    }

    pub fn antigeo(self, lhs: Blade, rhs: Blade) -> Blade {
        let lhs = self.left_comp(lhs);
        let rhs = self.left_comp(rhs);
        let output = self.geo(lhs, rhs);
        self.right_comp(output)
    }

    pub fn antidot(self, lhs: Blade, rhs: Blade) -> Blade {
        let lhs = self.left_comp(lhs);
        let rhs = self.left_comp(rhs);
        let output = self.dot(lhs, rhs);
        self.right_comp(output)
    }

    pub fn antiwedge(self, lhs: Blade, rhs: Blade) -> Blade {
        let lhs = self.left_comp(lhs);
        let rhs = self.left_comp(rhs);
        let output = self.wedge(lhs, rhs);
        self.right_comp(output)
    }

    pub fn wedge(self, lhs: Blade, rhs: Blade) -> Blade {
        let sum = lhs.grade() + rhs.grade();
        let product = self.geo(lhs, rhs);
        if product.grade() == sum {
            product
        } else {
            Blade::zero()
        }
    }

    pub fn antirev(self, blade: Blade) -> Blade {
        let comp = self.left_comp(blade);
        let rev = comp.rev();
        self.right_comp(rev)
    }

    pub fn blades_by_grade(self) -> impl Iterator<Item = Blade> {
        self.grade_range()
            .flat_map(move |g| self.blades().filter(move |b| b.grade() == g))
    }

    pub fn blades(self) -> Blades {
        let pseudoscalar = self.pseudoscalar();
        Blades {
            range: 0..=pseudoscalar.0,
        }
    }

    pub fn pseudoscalar(self) -> Blade {
        let not = u32::MAX << self.bases.len();
        Blade(!not)
    }

    pub fn right_comp(self, blade: Blade) -> Blade {
        if blade.is_zero() {
            return Blade::zero();
        }
        let i = self.pseudoscalar();
        let comp = i ^ blade;
        if self.geo(blade, comp).is_positive() {
            comp
        } else {
            -comp
        }
    }

    pub fn left_comp(self, blade: Blade) -> Blade {
        if blade.is_zero() {
            return Blade::zero();
        }
        let i = self.pseudoscalar();
        let comp = i ^ blade;
        if self.geo(comp, blade).is_positive() {
            comp
        } else {
            -comp
        }
    }

    pub fn symmetrical_complements(self) -> bool {
        self.blades()
            .all(|blade| self.left_comp(blade) == self.right_comp(blade))
    }

    pub fn grade(self, grade: u32) -> impl Iterator<Item = Blade> {
        self.blades().filter(move |b| b.grade() == grade)
    }

    pub fn grades(self) -> impl Iterator<Item = Type> {
        self.grade_range().map(Type::Grade)
    }

    pub fn grade_range(self) -> std::ops::RangeInclusive<u32> {
        0..=(self.bases.len() as u32)
    }

    pub fn types(self) -> impl Iterator<Item = Type> {
        let versors = [Type::Motor, Type::Flector, Type::Mv]
            .iter()
            .filter(move |ty| {
                let mut some = None;
                for blade in ty.iter_blades_unsorted(self) {
                    match some {
                        None => some = Some(blade.grade()),
                        Some(g) => {
                            if blade.grade() != g {
                                return true;
                            }
                        }
                    };
                }
                false
            })
            .copied();
        self.grades().chain(versors)
    }

    pub fn type_tuples(self) -> impl Iterator<Item = (Type, Type)> {
        self.types()
            .flat_map(move |lhs| self.types().map(move |rhs| (lhs, rhs)))
    }

    pub fn grade_tuples(self) -> impl Iterator<Item = (Type, Type)> {
        self.grades()
            .flat_map(move |lhs| self.grades().map(move |rhs| (lhs, rhs)))
    }
}

#[derive(Clone)]
pub struct Blades {
    range: std::ops::RangeInclusive<u32>,
}

impl Iterator for Blades {
    type Item = Blade;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(Blade)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Type {
    Grade(u32),
    Motor,
    Flector,
    Mv,
}

impl Type {
    pub fn name(self) -> &'static str {
        match self {
            Type::Grade(n) => match n {
                0 => "Scalar",
                1 => "Vector",
                2 => "Bivector",
                3 => "Trivector",
                4 => "Quadvector",
                5 => "Pentavector",
                _ => unimplemented!("grade out of range: {n}"),
            },
            Type::Motor => "Motor",
            Type::Flector => "Flector",
            Type::Mv => "Multivector",
        }
    }

    pub fn name_lowercase(self) -> &'static str {
        match self {
            Type::Grade(n) => match n {
                0 => "scalar",
                1 => "vector",
                2 => "bivector",
                3 => "trivector",
                4 => "quadvector",
                5 => "pentavector",
                _ => unimplemented!("grade out of range: {n}"),
            },
            Type::Motor => "motor",
            Type::Flector => "flector",
            Type::Mv => "multivector",
        }
    }

    pub fn iter_blades_sorted(self, algebra: Algebra) -> impl Iterator<Item = Blade> {
        algebra.blades_by_grade().filter(move |b| self.contains(*b))
    }

    pub fn single_blade(self, algebra: Algebra) -> bool {
        self.iter_blades_unsorted(algebra).nth(1).is_none()
    }

    pub fn iter_blades_unsorted(self, algebra: Algebra) -> IterBlades {
        IterBlades {
            blades: algebra.blades(),
            ty: self,
        }
    }

    pub fn contains(self, blade: Blade) -> bool {
        if blade.is_zero() {
            return false;
        }
        match self {
            Type::Grade(g) => blade.grade() == g,
            Type::Motor => blade.grade() & 1 != 1,
            Type::Flector => blade.grade() & 1 == 1,
            Type::Mv => true,
        }
    }

    pub fn complement(self, algebra: Algebra) -> Self {
        self.iter_blades_unsorted(algebra)
            .map(|blade| algebra.left_comp(blade))
            .collect::<Option<Self>>()
            .unwrap()
    }

    pub fn contains_ty(self, other: Self) -> bool {
        match (self, other) {
            (lhs, rhs) if lhs == rhs => true,
            (Type::Motor, Type::Grade(g)) => g & 1 != 1,
            (Type::Flector, Type::Grade(g)) => g & 1 == 1,
            (Type::Mv, _) => true,
            _ => false,
        }
    }
}

impl FromIterator<Blade> for Option<Type> {
    fn from_iter<T: IntoIterator<Item = Blade>>(iter: T) -> Self {
        let mut any = false;
        let mut all_even = true;
        let mut all_odd = true;
        let mut all_grade: Result<Option<u32>, ()> = Ok(None);

        for blade in iter {
            if blade.is_zero() {
                continue;
            }

            any = true;

            let grade = blade.grade();

            if grade & 1 == 1 {
                all_even = false;
            } else {
                all_odd = false;
            }

            all_grade = match all_grade {
                Ok(None) => Ok(Some(grade)),
                Ok(Some(g)) => {
                    if g == grade {
                        Ok(Some(g))
                    } else {
                        Err(())
                    }
                }
                Err(()) => Err(()),
            }
        }

        if let Ok(Some(grade)) = all_grade {
            return Some(Type::Grade(grade));
        }

        if !any {
            return None;
        }

        if all_even {
            return Some(Type::Motor);
        }

        if all_odd {
            return Some(Type::Flector);
        }

        Some(Type::Mv)
    }
}

#[derive(Clone)]
pub struct IterBlades {
    blades: Blades,
    ty: Type,
}

impl Iterator for IterBlades {
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

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Basis {
    pub char: char,
    pub sqr: Square,
}

impl Basis {
    fn sqr(self) -> Blade {
        self.sqr.blade()
    }
}

#[derive(Default, Copy, Clone, Eq, PartialEq)]
pub enum Square {
    #[default]
    Pos,
    Neg,
    Zero,
}

impl Square {
    fn blade(self) -> Blade {
        match self {
            Self::Pos => Blade(0),
            Self::Neg => -Blade(0),
            Self::Zero => Blade::zero(),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Blade(u32);

impl std::fmt::Debug for Blade {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Blade({:032b})", self.0)
    }
}

impl Blade {
    const SIGN: u32 = 1 << 31;
    const ZERO: u32 = 1 << 30;

    pub fn zero() -> Self {
        Self(Self::ZERO)
    }

    pub fn rev(self) -> Self {
        let count = self.unsigned().count();
        let half = count / 2;
        let odd = half & 1 == 1;
        if odd {
            -self
        } else {
            self
        }
    }

    pub fn unsigned(self) -> Self {
        Blade(self.0 & !Self::SIGN)
    }

    pub const fn is_positive(self) -> bool {
        self.0 & Self::SIGN != Self::SIGN
    }

    pub const fn is_negative(self) -> bool {
        !self.is_positive()
    }

    pub const fn is_zero(self) -> bool {
        self.0 & Self::ZERO == Self::ZERO
    }

    fn get_higher(self, i: u32) -> Self {
        let mask = u32::MAX << (i + 1);
        Self(self.unsigned().0 & mask)
    }

    fn get_lower(self, i: u32) -> Self {
        let mask = !(u32::MAX << i);
        Self(self.0 & mask)
    }

    fn count_higher(self, i: u32) -> u32 {
        self.get_higher(i).count()
    }

    fn count_lower(self, i: u32) -> u32 {
        self.get_lower(i).count()
    }

    fn count(self) -> u32 {
        self.unsigned().0.count_ones()
    }

    fn product(self, rhs: Self) -> Self {
        let is_zero = Blade(self.0 | rhs.0).is_zero();
        if is_zero {
            return Self::zero();
        }

        let l = self.unsigned();
        let mut r = rhs.unsigned();

        let start = r.0.trailing_zeros();
        let end = (std::mem::size_of::<Self>() * 8) as u32 - r.0.leading_zeros();

        let mut count = 0u32;
        for i in start..end {
            if r.contains(i) {
                r.flip(i);
                count += l.count_higher(i) + r.count_lower(i);
            }
        }

        let output = self ^ rhs;
        if count & 1 == 1 {
            -output
        } else {
            output
        }
    }

    fn contains(self, i: u32) -> bool {
        let flag = 1 << i;
        self.0 & flag == flag
    }

    fn flip(&mut self, i: u32) {
        let flag = 1 << i;
        self.0 ^= flag;
    }

    pub fn grade(self) -> u32 {
        self.unflagged().0.count_ones()
    }

    fn unflagged(self) -> Self {
        Blade(self.0 & !(Self::SIGN | Self::ZERO))
    }
}

impl std::ops::Neg for Blade {
    type Output = Self;
    fn neg(self) -> Self {
        Self(self.0 ^ Self::SIGN)
    }
}

impl std::ops::BitOr for Blade {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitXor for Blade {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl std::ops::BitAnd for Blade {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ProductOp {
    Geo,
    Dot,
    Wedge,
    Antigeo,
    Antidot,
    Antiwedge,
    Mul,
}

impl ProductOp {
    pub fn iter_local() -> impl Iterator<Item = Self> {
        IntoIterator::into_iter([
            Self::Geo,
            Self::Wedge,
            Self::Dot,
            Self::Antigeo,
            Self::Antidot,
            Self::Antiwedge,
        ])
    }

    pub fn iter_all() -> impl Iterator<Item = Self> {
        IntoIterator::into_iter([
            Self::Geo,
            Self::Wedge,
            Self::Dot,
            Self::Antigeo,
            Self::Antidot,
            Self::Antiwedge,
            Self::Mul,
        ])
    }

    pub fn output(self, algebra: Algebra, lhs: Type, rhs: Type) -> Option<Type> {
        cartesian_product(
            lhs.iter_blades_unsorted(algebra),
            rhs.iter_blades_unsorted(algebra),
        )
        .map(|(lhs, rhs)| self.product(algebra, lhs, rhs))
        .collect()
    }

    pub fn product(self, algebra: Algebra, lhs: Blade, rhs: Blade) -> Blade {
        match self {
            ProductOp::Geo | ProductOp::Mul => algebra.geo(lhs, rhs),
            ProductOp::Dot => algebra.dot(lhs, rhs),
            ProductOp::Wedge => algebra.wedge(lhs, rhs),
            ProductOp::Antigeo => algebra.antigeo(lhs, rhs),
            ProductOp::Antidot => algebra.antidot(lhs, rhs),
            ProductOp::Antiwedge => algebra.antiwedge(lhs, rhs),
        }
    }
}

pub fn cartesian_product<Lhs, Rhs>(
    lhs: Lhs,
    rhs: Rhs,
) -> impl Iterator<Item = (Lhs::Item, Rhs::Item)>
where
    Lhs: IntoIterator,
    <Lhs as IntoIterator>::Item: Copy,
    Rhs: IntoIterator + Clone,
{
    lhs.into_iter()
        .flat_map(move |lhs| rhs.clone().into_iter().map(move |rhs| (lhs, rhs)))
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SumOp {
    Add,
    Sub,
}

impl SumOp {
    pub fn iter() -> impl Iterator<Item = Self> {
        IntoIterator::into_iter([Self::Add, Self::Sub])
    }

    pub fn sum(algebra: Algebra, lhs: Type, rhs: Type) -> Option<Type> {
        lhs.iter_blades_unsorted(algebra)
            .chain(rhs.iter_blades_unsorted(algebra))
            .collect()
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Complement {
    Dual,
    LeftComp,
    RightComp,
}

impl Complement {
    pub fn iter(algebra: Algebra) -> impl Iterator<Item = Self> {
        let symmetric = algebra.symmetrical_complements();

        let dual = symmetric.then_some(Complement::Dual);
        let left_comp = (!symmetric).then_some(Complement::LeftComp);
        let right_comp = (!symmetric).then_some(Complement::RightComp);

        IntoIterator::into_iter([dual, left_comp, right_comp]).flatten()
    }

    pub fn call(self, algebra: Algebra, blade: Blade) -> Blade {
        match self {
            Self::Dual | Self::RightComp => algebra.right_comp(blade),
            Self::LeftComp => algebra.left_comp(blade),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ScalarOps {
    Mul,
    Div,
}

impl ScalarOps {
    pub fn iter() -> impl Iterator<Item = Self> {
        IntoIterator::into_iter([Self::Mul, Self::Div])
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ScalarAssignOps {
    Mul,
    Div,
}

impl ScalarAssignOps {
    pub fn iter() -> impl Iterator<Item = Self> {
        IntoIterator::into_iter([Self::Mul, Self::Div])
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SumAssignOps {
    AddAssign,
    SubAssign,
}

impl SumAssignOps {
    pub fn iter() -> impl Iterator<Item = Self> {
        IntoIterator::into_iter([Self::AddAssign, Self::SubAssign])
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum FloatConversion {
    ToF32,
    ToF64,
}

impl FloatConversion {
    pub fn iter() -> impl Iterator<Item = Self> {
        IntoIterator::into_iter([Self::ToF32, Self::ToF64])
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum NormOps {
    Norm2,
    Norm,
}

impl NormOps {
    pub fn iter() -> impl Iterator<Item = Self> {
        IntoIterator::into_iter([Self::Norm2, Self::Norm])
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum InverseOps {
    Inverse,
    Unitize,
}

impl InverseOps {
    pub fn iter() -> impl Iterator<Item = Self> {
        IntoIterator::into_iter([Self::Inverse, Self::Unitize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos_and_neg_scalar() {
        let s = Blade(0);
        assert_eq!(s, s);
        assert_eq!(-s, -s);
        assert_ne!(-s, s);
    }

    #[test]
    fn scalar_closed_mul() {
        let a = Algebra::ga2();
        let s = Blade(0);
        assert_eq!(s, a.geo(s, s));

        assert!(s.is_positive());
        assert!(!s.is_negative());
    }

    #[test]
    fn vector_mul_scalar() {
        let a = Algebra::ga3();
        let s = Blade(0);
        let e1 = Blade(0b001);
        let e2 = Blade(0b010);
        let e3 = Blade(0b100);

        assert_eq!(s, a.geo(e1, e1));
        assert_eq!(s, a.geo(e2, e2));
        assert_eq!(s, a.geo(e3, e3));
    }

    #[test]
    fn blade_set_count() {
        assert_eq!(0, Blade(0).count());
        assert_eq!(0, (-Blade(0)).count());
        assert_eq!(1, Blade(1).count());
        assert_eq!(1, (-Blade(1)).count());
        assert_eq!(2, Blade(0b11).count());
        assert_eq!(2, (-Blade(0b11)).count());
        assert_eq!(3, Blade(0b111).count());
        assert_eq!(3, (-Blade(0b111)).count());
    }

    #[test]
    fn blade_set_get_higher() {
        assert_eq!(Blade(0), Blade(0).get_higher(0));
        assert_eq!(Blade(0), Blade(0b1).get_higher(0));
        assert_eq!(Blade(0), Blade(0b1).get_higher(1));
        assert_eq!(Blade(0b10), Blade(0b11).get_higher(0));
        assert_eq!(Blade(0), Blade(0b11).get_higher(1));
        assert_eq!(Blade(0), Blade(0b11).get_higher(2));
        assert_eq!(Blade(0b110), Blade(0b111).get_higher(0));
        assert_eq!(Blade(0b100), Blade(0b111).get_higher(1));
        assert_eq!(Blade(0), Blade(0b111).get_higher(2));
    }

    #[test]
    fn blade_set_count_higher() {
        assert_eq!(0, Blade(0).count_higher(0));
        assert_eq!(0, Blade(0b1).count_higher(0));
        assert_eq!(0, Blade(0b1).count_higher(1));
        assert_eq!(1, Blade(0b11).count_higher(0));
        assert_eq!(0, Blade(0b11).count_higher(1));
        assert_eq!(0, Blade(0b11).count_higher(2));
        assert_eq!(2, Blade(0b111).count_higher(0));
        assert_eq!(1, Blade(0b111).count_higher(1));
        assert_eq!(0, Blade(0b111).count_higher(2));
    }

    #[test]
    fn blade_set_get_lower() {
        assert_eq!(Blade(0), Blade(0).get_lower(0));
        assert_eq!(Blade(0), Blade(1).get_lower(0));
        assert_eq!(Blade(1), Blade(1).get_lower(1));
        assert_eq!(Blade(0), Blade(0b11).get_lower(0));
        assert_eq!(Blade(1), Blade(0b11).get_lower(1));
        assert_eq!(Blade(0b11), Blade(0b11).get_lower(2));
    }

    #[test]
    fn blade_set_count_lower() {
        assert_eq!(0, Blade(0).count_lower(0));
        assert_eq!(0, Blade(1).count_lower(0));
        assert_eq!(1, Blade(1).count_lower(1));
        assert_eq!(0, Blade(0b11).count_lower(0));
        assert_eq!(1, Blade(0b11).count_lower(1));
        assert_eq!(2, Blade(0b11).count_lower(2));
    }

    #[test]
    fn bivector_mul_neg_scalar() {
        let a = Algebra::ga3();
        let s = Blade(0);
        let e12 = Blade(0b011);
        let e23 = Blade(0b110);
        let e31 = -Blade(0b101);

        assert_eq!(-s, a.geo(e12, e12));
        assert_eq!(-s, a.geo(e23, e23));
        assert_eq!(-s, a.geo(e31, e31));
    }

    #[test]
    fn bivector_mul_e12_e23() {
        let a = Algebra::ga3();
        let e12 = Blade(0b011);
        let e23 = Blade(0b110);
        let e13 = Blade(0b101);

        assert_eq!(e13, a.geo(e12, e23));
    }

    #[test]
    fn bivector_mul_e23_e12() {
        let a = Algebra::ga3();
        let e12 = Blade(0b011);
        let e23 = Blade(0b110);
        let e13 = Blade(0b101);

        assert_eq!(-e13, a.geo(e23, e12));
    }

    #[test]
    fn unsigned() {
        let s = Blade(0);
        assert_eq!(s, s.unsigned());
        assert_eq!(s, (-s).unsigned());
    }

    #[test]
    fn rev_blade_set() {
        let s = Blade(0);
        let e1 = Blade(1);
        let e12 = Blade(0b11);
        let e123 = Blade(0b111);
        let e1234 = Blade(0b1111);
        let e12345 = Blade(0b11111);

        assert_eq!(s, s.rev());
        assert_eq!(e1, e1.rev());
        assert_eq!(-e12, e12.rev());
        assert_eq!(-e123, e123.rev());
        assert_eq!(e1234, e1234.rev());
        assert_eq!(e12345, e12345.rev());
    }

    #[test]
    fn blade_set_contains() {
        assert!(!Blade(0).contains(0));
        assert!(Blade(1).contains(0));
        assert!(!Blade(1).contains(1));
        assert!(Blade(0b11).contains(1));
    }

    #[test]
    fn vector_wedge_bivector() {
        let a = Algebra::ga3();
        let e12 = Blade(0b11);
        let e3 = Blade(0b100);
        let e123 = Blade(0b111);

        assert_eq!(e123, a.geo(e12, e3));
        assert_eq!(e123, a.geo(e3, e12));
    }

    #[test]
    fn blade_set_zero() {
        assert!(Blade::zero().is_zero());
        assert!(!Blade(0).is_zero());
        assert!(!Blade(1).is_zero());
    }

    #[test]
    fn blade_set_mul_zero() {
        let a = Algebra::ga2();
        assert_eq!(Blade::zero(), a.geo(Blade(1), Blade::zero()));
        assert_eq!(Blade::zero(), a.geo(Blade::zero(), Blade(1)));
    }

    #[test]
    fn zero_unsigned_is_zero() {
        assert_eq!(Blade::zero(), Blade::zero().unsigned());
    }

    #[test]
    fn dot_vectors() {
        let a = Algebra::ga2();
        let s = Blade(0);
        let e1 = Blade(1);
        let e2 = Blade(0b10);

        assert!(a.dot(e1, e2).is_zero());
        assert_eq!(s, a.dot(e1, e1));
    }

    #[test]
    fn wedge_vectors() {
        let a = Algebra::ga2();
        let e12 = Blade(0b11);
        let e1 = Blade(1);
        let e2 = Blade(0b10);

        assert!(a.wedge(e1, e1).is_zero());
        assert_eq!(e12, a.wedge(e1, e2));
        assert_eq!(-e12, a.wedge(e2, e1));
    }

    #[test]
    fn mul_zero_sqr() {
        let a = Algebra {
            bases: &[Basis {
                char: 'n',
                sqr: Square::Zero,
            }],
        };
        let e1 = Blade(1);

        assert!(a.geo(e1, e1).is_zero());
    }

    #[test]
    fn mul_neg_sqr() {
        let a = Algebra {
            bases: &[Basis {
                char: 'n',
                sqr: Square::Neg,
            }],
        };
        let e1 = Blade(1);
        let neg_s = -Blade(0);

        assert_eq!(neg_s, a.geo(e1, e1));
    }

    #[test]
    fn type_from_scalar() {
        let type_ = Option::<Type>::from_iter([Blade(0)]).unwrap();
        assert_eq!(Type::Grade(0), type_);
    }

    #[test]
    fn type_from_vector() {
        let type_ = Option::<Type>::from_iter([Blade(1), Blade(0b10), Blade(0b100)]).unwrap();
        assert_eq!(Type::Grade(1), type_);
    }

    #[test]
    fn type_from_bivector() {
        let type_ = Option::<Type>::from_iter([Blade(0b11), Blade(0b110), Blade(0b101)]).unwrap();
        assert_eq!(Type::Grade(2), type_);
    }

    #[test]
    fn type_from_trivector() {
        let type_ = Option::<Type>::from_iter([Blade(0b111)]).unwrap();
        assert_eq!(Type::Grade(3), type_);
    }

    #[test]
    fn type_from_motor() {
        let type_ = Option::<Type>::from_iter([Blade(0), Blade(0b11)]).unwrap();
        assert_eq!(Type::Motor, type_);
    }

    #[test]
    fn type_from_flector() {
        let type_ = Option::<Type>::from_iter([Blade(0b1), Blade(0b111)]).unwrap();
        assert_eq!(Type::Flector, type_);
    }

    #[test]
    fn type_from_none() {
        let type_ = Option::<Type>::from_iter([]);
        assert_eq!(None, type_);
    }

    #[test]
    fn algebra_grades() {
        let a = Algebra::ga2();
        assert_eq!(3, a.grades().count());

        let a = Algebra::cga3();
        assert_eq!(6, a.grades().count());
    }

    #[test]
    fn motor_contains() {
        assert!(Type::Motor.contains(Blade(0)));
        assert!(!Type::Motor.contains(Blade(1)));
        assert!(Type::Motor.contains(Blade(0b11)));
        assert!(!Type::Motor.contains(Blade(0b111)));

        assert!(!Type::Flector.contains(Blade(0)));
        assert!(Type::Flector.contains(Blade(1)));
        assert!(!Type::Flector.contains(Blade(0b11)));
        assert!(Type::Flector.contains(Blade(0b111)));

        assert!(Type::Grade(0).contains(Blade(0)));
        assert!(!Type::Grade(0).contains(Blade(1)));
    }

    #[test]
    fn skip_versors_with_one_grade() {
        let types = Algebra::ga2()
            .types()
            .collect::<std::collections::HashSet<_>>();

        assert!(types.contains(&Type::Motor));
        assert!(!types.contains(&Type::Flector));
    }

    #[test]
    fn add_scalar_bivector_to_motor() {
        let algebra = Algebra::ga3();

        assert_eq!(
            Some(Type::Motor),
            SumOp::sum(algebra, Type::Grade(0), Type::Grade(2))
        );
    }

    #[test]
    fn right_comp() {
        let a = Algebra::pga3();
        let i = a.pseudoscalar();
        for blade in a.blades() {
            let comp = a.right_comp(blade);
            assert_eq!(i, a.geo(blade, comp));
        }
    }

    #[test]
    fn symmetrical_complements() {
        let g3 = Algebra::ga3();
        let pga3 = Algebra::pga3();

        assert!(g3.symmetrical_complements());
        assert!(!pga3.symmetrical_complements());
    }

    #[test]
    fn complements() {
        let a = Algebra::ga3();
        assert_eq!(Type::Grade(2), Type::Grade(1).complement(a));
        assert_eq!(Type::Grade(1), Type::Grade(2).complement(a));
        assert_eq!(Type::Flector, Type::Motor.complement(a));
        assert_eq!(Type::Motor, Type::Flector.complement(a));
    }

    #[test]
    fn type_does_not_contain_zero() {
        assert!(!Type::Grade(0).contains(Blade::zero()));
    }

    #[test]
    #[allow(non_snake_case)]
    fn antidot() {
        let a = Algebra::pga3();

        let s = Blade(0);
        let I = Blade(0b1111);
        let e1 = Blade(0b0001);
        let e4 = Blade(0b1000);
        let e124 = Blade(0b1011);
        let e123 = Blade(0b0111);

        assert_eq!(e123, a.antidot(e4, s));
        assert_eq!(-I, a.antidot(e4, e4));
        assert_eq!(I, a.antidot(e124, e124));
        assert_eq!(I, a.antidot(I, I));
        assert_eq!(Blade::zero(), a.antidot(e1, e1));
    }

    #[test]
    #[allow(non_snake_case)]
    fn antiwedge() {
        let a = Algebra::pga3();

        let z = Blade::zero();
        let s = Blade(0);
        let e1 = Blade(0b0001);
        let e4 = Blade(0b1000);
        let e14 = Blade(0b1001);
        let e23 = Blade(0b0110);
        let e234 = Blade(0b1110);

        assert_eq!(s, a.antiwedge(e14, e23));
        assert_eq!(-s, a.antiwedge(e234, e1));
        assert_eq!(s, a.antiwedge(e1, e234));
        assert_eq!(z, a.antiwedge(e1, e1));
        assert_eq!(z, a.antiwedge(e1, e4));
    }

    #[test]
    fn antirev() {
        let a = Algebra::pga3();
        let s = Blade(0);
        let e1 = Blade(1);
        let e12 = Blade(0b11);
        let e123 = Blade(0b111);
        let e1234 = Blade(0b1111);
        assert_eq!(s, a.antirev(s));
        assert_eq!(-e1, a.antirev(e1));
        assert_eq!(-e12, a.antirev(e12));
        assert_eq!(e123, a.antirev(e123));
        assert_eq!(e1234, a.antirev(e1234));
    }
}

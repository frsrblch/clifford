use crate::IsEven;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Blade(pub u32);

impl std::fmt::Debug for Blade {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let sign = if self.is_positive() { '+' } else { '-' };
        write!(f, "Blade({sign}{:06b})", self.unsigned().0)
    }
}

impl Ord for Blade {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.unsigned().0.cmp(&other.unsigned().0) {
            std::cmp::Ordering::Equal => match (self.is_negative(), other.is_negative()) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (true, true) | (false, false) => std::cmp::Ordering::Equal,
            },
            cmp => cmp,
        }
    }
}

impl PartialOrd for Blade {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Blade {
    const SIGN: u32 = 1 << 31;
    const ZERO: u32 = 1 << 30;

    pub fn scalar() -> Self {
        Self(0)
    }

    pub fn zero() -> Self {
        Self(Self::ZERO)
    }

    pub fn filter<F: FnOnce(Self) -> bool>(self, f: F) -> Blade {
        if self.is_zero() {
            return self;
        }

        if f(self) {
            self
        } else {
            Blade::zero()
        }
    }

    pub fn rev(self) -> Self {
        let count = self.count();
        let half = count / 2;
        let rev = half & 1 == 1;
        if rev {
            -self
        } else {
            self
        }
    }

    pub fn grade_involution(self) -> Self {
        let odd = self.count() & 1 == 1;
        if odd {
            -self
        } else {
            self
        }
    }

    pub fn clifford_conjugate(self) -> Self {
        let count = self.count();
        let half = count / 2;
        let rev = half & 1 == 1;
        let odd = self.count() & 1 == 1;
        if rev ^ odd {
            -self
        } else {
            self
        }
    }

    pub fn unsigned(self) -> Self {
        Blade(self.0 & !Self::SIGN)
    }

    pub fn sign(self) -> Self {
        Blade(self.0 & Self::SIGN)
    }

    pub const fn is_positive(self) -> bool {
        self.0 & Self::SIGN != Self::SIGN && !self.is_zero()
    }

    pub const fn is_negative(self) -> bool {
        self.0 & Self::SIGN == Self::SIGN && !self.is_zero()
    }

    pub const fn is_zero(self) -> bool {
        self.0 & Self::ZERO == Self::ZERO
    }

    pub fn is_scalar(self) -> bool {
        self == Blade(0)
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

    pub fn product(self, rhs: Self) -> Self {
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
        if count.is_even() {
            output
        } else {
            -output
        }
    }

    pub fn contains(self, i: u32) -> bool {
        let flag = 1 << i;
        self.0 & flag == flag
    }

    pub fn flip(&mut self, i: u32) {
        let flag = 1 << i;
        self.0 ^= flag;
    }

    pub fn grade(self) -> u32 {
        self.unflagged().0.count_ones()
    }

    fn unflagged(self) -> Self {
        Blade(self.0 & !(Self::SIGN | Self::ZERO))
    }

    pub fn pseudoscalar(dim: u32) -> Self {
        let inverse = u32::MAX << dim;
        Blade(!inverse)
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

impl<T> std::ops::Index<Blade> for Vec<T> {
    type Output = T;

    fn index(&self, index: Blade) -> &Self::Output {
        self.index(index.unsigned().0 as usize)
    }
}

impl<T> std::ops::IndexMut<Blade> for Vec<T> {
    fn index_mut(&mut self, index: Blade) -> &mut Self::Output {
        self.index_mut(index.unsigned().0 as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::Blade;

    #[test]
    fn pos_and_neg_scalar() {
        let s = Blade(0);
        assert_eq!(s, s);
        assert_eq!(-s, -s);
        assert_ne!(-s, s);
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
    fn unsigned() {
        let s = Blade::scalar();
        assert_eq!(s, s.unsigned());
        assert_eq!(s, (-s).unsigned());

        let z = Blade::zero();
        assert_eq!(z, z.unsigned());
        assert_eq!(z, (-z).unsigned());
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
    fn blade_set_zero() {
        assert!(Blade::zero().is_zero());
        assert!(!Blade(0).is_zero());
        assert!(!Blade(1).is_zero());
    }

    #[test]
    fn blade_grade() {
        let s = Blade::scalar();
        assert_eq!(0, s.grade());
        assert_eq!(0, (-s).grade());
    }

    #[test]
    fn blade_zero_grade() {
        let z = Blade::zero();
        assert_eq!(0, z.grade());
        assert_eq!(0, (-z).grade());
    }

    #[test]
    fn grade_involution() {
        assert_eq!(Blade::scalar(), Blade::scalar().grade_involution());
        assert_eq!(-Blade(0b1), Blade(0b1).grade_involution());
        assert_eq!(Blade(0b11), Blade(0b11).grade_involution());
        assert_eq!(-Blade(0b111), Blade(0b111).grade_involution());
        assert_eq!(Blade(0b1111), Blade(0b1111).grade_involution());
    }

    #[test]
    fn clifford_conjugate() {
        assert_eq!(Blade::scalar(), Blade::scalar().clifford_conjugate());
        assert_eq!(-Blade(0b1), Blade(0b1).clifford_conjugate());
        assert_eq!(-Blade(0b11), Blade(0b11).clifford_conjugate());
        assert_eq!(Blade(0b111), Blade(0b111).clifford_conjugate());
        assert_eq!(Blade(0b1111), Blade(0b1111).clifford_conjugate());
    }
}

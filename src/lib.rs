use std::collections::HashSet;

mod define;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Algebra {
    one: u8,
    zero: u8,
    neg_one: u8,
}

impl Algebra {
    pub fn new(one: u8, zero: u8, neg_one: u8) -> Self {
        Self { one, zero, neg_one }
    }

    pub fn get(&self, index: u8) -> Basis {
        Basis(index, *self)
    }

    pub fn square(&self, index: u8) -> Multiplier {
        if index == 0 {
            panic!("zero is not a valid index");
        }

        if index <= self.one {
            Multiplier::One
        } else if index <= self.one + self.zero {
            Multiplier::Zero
        } else if index <= self.one + self.zero + self.neg_one {
            Multiplier::NegOne
        } else {
            panic!("index out of range: {index}");
        }
    }

    pub fn bases(&self) -> impl Iterator<Item = Basis> + '_ {
        (1..=self.sum()).into_iter().map(|i| self.get(i))
    }

    pub fn grades(&self) -> impl Iterator<Item = Grade> + '_ {
        (1..self.sum()).into_iter().map(|i| Grade(i, *self))
    }

    pub fn psuedovector(&self) -> BladeSet {
        let mut set = BladeSet(0);
        for i in 1..=self.sum() {
            set.insert(i);
        }
        set
    }

    pub fn grade(&self, grade: u8) -> Grade {
        match grade {
            grade if grade <= self.sum() => Grade(grade, *self),
            _ => panic!("invalid grade: {grade}"),
        }
    }

    pub fn blades(&self) -> impl Iterator<Item = Blade> + '_ {
        (0..=self.psuedovector().0)
            .into_iter()
            .map(|set| Blade(BladeSet(set), *self))
    }

    fn sum(&self) -> u8 {
        self.one + self.zero + self.neg_one
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Multiplier {
    One,
    Zero,
    NegOne,
}

impl std::ops::Mul for Multiplier {
    type Output = Multiplier;

    fn mul(self, rhs: Self) -> Self::Output {
        use Multiplier::*;
        match (self, rhs) {
            (Zero, _) | (_, Zero) => Zero,
            (One, One) | (NegOne, NegOne) => One,
            (One, NegOne) | (NegOne, One) => NegOne,
        }
    }
}

impl std::ops::MulAssign for Multiplier {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Basis(u8, Algebra);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Grade(u8, Algebra);

impl Grade {
    fn blades(&self) -> impl Iterator<Item = Blade> + '_ {
        self.1.blades().filter(|b| b.0.len() == self.0)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Blade(BladeSet, Algebra);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct BladeSet(u32);

impl BladeSet {
    pub fn contains(&self, index: u8) -> bool {
        assert!(index > 0, "index cannot be zero (e1 is stored at index 0)");
        let flag = 1 << (index - 1);
        self.0 & flag == flag
    }

    pub fn insert(&mut self, index: u8) {
        assert!(index > 0, "index cannot be zero (e1 is stored at index 0)");
        self.0 |= 1 << (index - 1);
    }

    pub fn flip(&mut self, index: u8) {
        let flag = 1 << (index - 1);
        self.0 ^= flag;
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub fn len(&self) -> u8 {
        self.0.count_ones() as u8
    }
}

impl std::ops::Mul for Blade {
    type Output = (Multiplier, Blade);

    fn mul(mut self, mut rhs: Self) -> Self::Output {
        assert_eq!(self.1, rhs.1, "different algebras");

        let mut multiplier = Multiplier::One;

        for base in self.1.bases() {
            if rhs.0.contains(base.0) {
                let swaps = ((base.0 + 1)..=self.1.sum())
                    .into_iter()
                    .filter(|&b| self.0.contains(b))
                    .count()
                    + (1..base.0)
                        .into_iter()
                        .filter(|&b| rhs.0.contains(b))
                        .count();

                if swaps % 2 == 1 {
                    multiplier *= Multiplier::NegOne;
                }

                if self.0.contains(base.0) {
                    multiplier *= self.1.square(base.0);
                }
                self.0.flip(base.0);
                rhs.0.flip(base.0);
            }
        }

        debug_assert!(rhs.0.is_empty());

        (multiplier, self)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SubAlgebra {
    Even(Algebra),
    Odd(Algebra),
}

impl SubAlgebra {
    pub fn opposite(&self) -> Self {
        match self {
            SubAlgebra::Even(a) => SubAlgebra::Odd(*a),
            SubAlgebra::Odd(a) => SubAlgebra::Even(*a),
        }
    }

    pub fn blades(&self) -> impl Iterator<Item = Blade> + '_ {
        let filter = match self {
            SubAlgebra::Even(_) => |b: &Blade| b.0.len() % 2 == 0,
            SubAlgebra::Odd(_) => |b: &Blade| b.0.len() % 2 != 0,
        };
        self.algebra().blades().filter(filter)
    }

    fn algebra(&self) -> &Algebra {
        match self {
            SubAlgebra::Even(a) => a,
            SubAlgebra::Odd(a) => a,
        }
    }
}

pub enum Type {
    Zero,
    Grade(Grade),
    SubAlgebra(SubAlgebra),
}

impl FromIterator<Blade> for Type {
    fn from_iter<T: IntoIterator<Item = Blade>>(iter: T) -> Self {
        let mut all_even = true;
        let mut all_odd = true;
        let mut grades = HashSet::new();
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
            return Type::Zero;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(debug_assertions))]
    fn write_to_output_file() {
        let path = std::path::Path::new("output.txt");
        let _ = std::fs::write(path, Algebra::new(3, 0, 0).define().to_string());
    }

    #[test]
    #[should_panic]
    fn no_bases() {
        Algebra::new(0, 0, 0).square(0);
    }

    #[test]
    fn one_d() {
        let bases = Algebra::new(1, 0, 0);
        assert_eq!(Multiplier::One, bases.square(1));
        assert!(std::panic::catch_unwind(|| bases.square(2)).is_err());
    }

    #[test]
    fn pga() {
        let bases = Algebra::new(3, 1, 0);
        assert_eq!(Multiplier::One, bases.square(3));
        assert_eq!(Multiplier::Zero, bases.square(4));
        assert!(std::panic::catch_unwind(|| bases.square(5)).is_err());
    }

    #[test]
    fn cga() {
        let bases = Algebra::new(4, 0, 1);
        assert_eq!(Multiplier::One, bases.square(4));
        assert_eq!(Multiplier::NegOne, bases.square(5));
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
        let g3 = Algebra::new(3, 0, 0);

        assert_eq!(BladeSet(0b_0111), g3.psuedovector());
    }

    #[test]
    fn grade_blades() {
        let bases = Algebra::new(3, 1, 0);

        assert_eq!(1, bases.grade(0).blades().count());
        assert_eq!(4, bases.grade(1).blades().count());
        assert_eq!(6, bases.grade(2).blades().count());
        assert_eq!(4, bases.grade(3).blades().count());
        assert_eq!(1, bases.grade(4).blades().count());
    }

    #[test]
    fn blade_multiplication() {
        let alg = Algebra::new(3, 1, 1);
        let e12 = Blade(BladeSet(0b_0011), alg);
        let e23 = Blade(BladeSet(0b_0110), alg);
        let e24 = Blade(BladeSet(0b_1010), alg);
        let e5 = Blade(BladeSet(0b_1_0000), alg);

        assert_eq!((Multiplier::NegOne, Blade(BladeSet(0), alg)), e12 * e12);
        assert_eq!((Multiplier::One, Blade(BladeSet(0b_0101), alg)), e12 * e23);
        assert_eq!((Multiplier::One, Blade(BladeSet(0b_1001), alg)), e12 * e24);
        assert_eq!(Multiplier::Zero, (e24 * e24).0);
        assert_eq!((Multiplier::NegOne, Blade(BladeSet(0), alg)), e5 * e5);
    }
}

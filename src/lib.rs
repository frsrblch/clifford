#![feature(
    const_trait_impl,
    const_fn_floating_point_arithmetic,
    const_default_impls,
    const_convert
)]

// TODO create proc_macro crate
// TODO grade sub, heterogeneous add/sub,reverses, complements, antiproducts

mod define;
//
// /// Point at the origin
// const N_BAR: Vector = Vector::new(0., 0., 0., 1., -1.);
//
// /// Point through infinity
// const N: Vector = Vector::new(0., 0., 0., 1., 1.);
//
// /// GA for Physicists, Chapter 10
// fn point(x: f64, y: f64, z: f64) -> Vector {
//     let x2 = x * x + y * y + z * z;
//     let x = Vector::new(x, y, z, 0., 0.);
//     x2 * N + 2.0 * x - N_BAR
// }
//
// /// Check whether point passes through infinity
// fn is_flat<T, W>(value: T) -> bool
// where
//     Vector: Wedge<T, Output = W>,
//     W: PartialEq + From<Zero>,
// {
//     N.wedge(value) == W::from(Zero)
// }
//
// #[test]
// fn null_squares() {
//     assert_eq!(N_BAR * N_BAR, Even::default());
//
//     assert_eq!(N * N, Even::default());
//
//     assert_eq!(N_BAR.dot(N), 2.);
//
//     assert_eq!(
//         N_BAR.wedge(N),
//         Bivector::new(0., 0., 0., 0., 0., 0., 0., 0., 0., 2.0)
//     );
// }
//
// #[test]
// fn point_pair() {
//     let pt = point(0., 0., 0.);
//     let b = point(1., 0., 0.);
//     let c = point(0., 1., 0.);
//     let d = point(0., 0., 1.);
//
//     assert_eq!(pt.dot(b), -2.);
//     assert!(!is_flat(pt));
//
//     let pair = pt.wedge(b);
//     assert!(!is_flat(pair));
//
//     let line = pair.wedge(N_BAR);
//     assert!(is_flat(line));
//
//     let plane = line.wedge(c);
//     assert!(is_flat(plane));
//
//     let circle = pair.wedge(c);
//     assert!(!is_flat(circle));
//
//     let sphere = circle.wedge(d);
//     assert!(!is_flat(sphere));
// }

// #[test]
// fn vec_mul() {
//
//
//     let a = Vector::new(2., 3., 4., 1.);
//     let b = Vector::new(3., 5., 7., 1.);
//     let c = Vector::new(0., 0., 0., 1.);
//
//     let line = a.wedge(b);
//     let plane = line.wedge(c);
//
//     dbg!(line, plane);
//     panic!("done");
// }

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

    pub fn basis(&self, index: u8) -> Basis {
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

    pub fn types(&self) -> Vec<Type> {
        Type::iter(*self)
    }

    pub fn bases(&self) -> impl Iterator<Item = Basis> + '_ {
        (1..=self.sum()).into_iter().map(|i| self.basis(i))
    }

    pub fn grades_without_scalar(&self) -> impl Iterator<Item = Grade> + '_ {
        (1..=self.sum()).into_iter().map(|i| self.grade(i))
    }

    pub fn grades_with_scalar(&self) -> impl Iterator<Item = Grade> + '_ {
        (0..=self.sum()).into_iter().map(|i| self.grade(i))
    }

    pub fn psuedovector(&self) -> Blade {
        let mut set = BladeSet(0);
        for i in 1..=self.sum() {
            set.insert(i);
        }
        self.blade(set.0)
    }

    pub fn scalar(&self) -> Blade {
        self.blade(0)
    }

    pub fn grade(&self, grade: u8) -> Grade {
        match grade {
            grade if grade <= self.sum() => Grade(grade, *self),
            _ => panic!("invalid grade: {grade}"),
        }
    }

    pub fn blades(&self) -> impl Iterator<Item = Blade> + '_ {
        self.grades_with_scalar().flat_map(|grade| {
            self.blades_unsorted()
                .filter(move |blade| blade.grade() == grade)
        })
    }

    pub fn subalgebras(&self) -> impl Iterator<Item = SubAlgebra> + '_ {
        [SubAlgebra::Even(*self), SubAlgebra::Odd(*self)].into_iter()
    }

    fn blades_unsorted(&self) -> impl Iterator<Item = Blade> + '_ {
        (0..=self.psuedovector().0 .0)
            .into_iter()
            .map(|set| self.blade(set))
    }

    pub fn blade(&self, set: u32) -> Blade {
        Blade(BladeSet(set), *self)
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
        self.1.blades_unsorted().filter(|b| b.0.len() == self.0)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Blade(BladeSet, Algebra);

impl Blade {
    pub fn grade(&self) -> Grade {
        Grade(self.0.len(), self.1)
    }

    pub fn dot(self, rhs: Self) -> (Multiplier, Blade) {
        let (multiplier, blade) = self * rhs;

        if multiplier == Multiplier::Zero {
            return (multiplier, blade);
        }

        let a = self.0.len();
        let b = rhs.0.len();
        let max = a.max(b);
        let min = a.min(b);
        let grade = max - min;

        if blade.0.len() == grade {
            (multiplier, blade)
        } else {
            (Multiplier::Zero, blade)
        }
    }

    pub fn wedge(self, rhs: Self) -> (Multiplier, Blade) {
        let (multiplier, blade) = self * rhs;

        if multiplier == Multiplier::Zero {
            return (multiplier, blade);
        }

        let a = self.0.len();
        let b = rhs.0.len();
        let grade = a + b;

        if blade.0.len() == grade {
            (multiplier, blade)
        } else {
            (Multiplier::Zero, blade)
        }
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Type {
    Zero(Algebra),
    Grade(Grade),
    SubAlgebra(SubAlgebra),
}

impl Type {
    fn from_iter<T: IntoIterator<Item = Blade>>(iter: T, alg: Algebra) -> Self {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(debug_assertions))]
    fn write_to_output_file() {
        let path = std::path::Path::new("output.txt");
        let _ = std::fs::write(path, Algebra::new(4, 0, 1).define().to_string());
    }

    #[test]
    #[should_panic]
    fn no_bases() {
        Algebra::new(0, 0, 0).square(0);
    }

    #[test]
    fn one_d() {
        let alg = Algebra::new(1, 0, 0);
        assert_eq!(Multiplier::One, alg.square(1));
        assert!(std::panic::catch_unwind(|| alg.square(2)).is_err());
    }

    #[test]
    fn pga() {
        let alg = Algebra::new(3, 1, 0);
        assert_eq!(Multiplier::One, alg.square(3));
        assert_eq!(Multiplier::Zero, alg.square(4));
        assert!(std::panic::catch_unwind(|| alg.square(5)).is_err());
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
        let alg = Algebra::new(3, 0, 0);

        assert_eq!(alg.blade(0b_0111), alg.psuedovector());
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

        assert_eq!((Multiplier::NegOne, alg.blade(0)), e12 * e12);
        assert_eq!((Multiplier::One, alg.blade(0b_0101)), e12 * e23);
        assert_eq!((Multiplier::One, alg.blade(0b_1001)), e12 * e24);
        assert_eq!(Multiplier::Zero, (e24 * e24).0);
        assert_eq!((Multiplier::NegOne, alg.blade(0)), e5 * e5);
    }

    #[test]
    fn blade_dot() {
        let alg = Algebra::new(3, 1, 0);
        let e12 = alg.blade(0b_0011);
        let e23 = alg.blade(0b_0110);
        let e34 = alg.blade(0b_1100);

        assert_eq!((Multiplier::NegOne, alg.blade(0)), e12.dot(e12));
        assert_eq!((Multiplier::Zero, alg.blade(0b_0101)), e12.dot(e23));
        assert_eq!((Multiplier::Zero, alg.blade(0b_1111)), e12.dot(e34));
    }

    #[test]
    fn blade_wedge() {
        let alg = Algebra::new(3, 1, 0);
        let e12 = alg.blade(0b_0011);
        let e23 = alg.blade(0b_0110);
        let e34 = alg.blade(0b_1100);

        assert_eq!((Multiplier::Zero, alg.blade(0)), e12.wedge(e12));
        assert_eq!((Multiplier::Zero, alg.blade(0b_0101)), e12.wedge(e23));
        assert_eq!((Multiplier::One, alg.blade(0b_1111)), e12.wedge(e34));
    }
}

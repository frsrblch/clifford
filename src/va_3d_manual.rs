use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

pub trait ScalarProduct<Rhs> {
    type Output;
    fn sp(self, rhs: Rhs) -> Self::Output;
}

pub trait BivectorProduct<Rhs> {
    type Output;
    fn bp(self, rhs: Rhs) -> Self::Output;
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Zero;

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Scalar<T, U> {
    pub value: T,
    marker: PhantomData<U>,
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Vector<T, U> {
    pub e1: T,
    pub e2: T,
    pub e3: T,
    marker: PhantomData<U>,
}

impl<T, U> Vector<T, U> {
    pub fn new(e1: T, e2: T, e3: T) -> Self {
        Vector {
            e1,
            e2,
            e3,
            marker: PhantomData,
        }
    }
}

impl<U> Vector<f64, U> {
    pub fn f32(self) -> Vector<f32, U> {
        Vector {
            e1: self.e1 as f32,
            e2: self.e2 as f32,
            e3: self.e3 as f32,
            marker: PhantomData,
        }
    }
}

impl<U> Vector<f32, U> {
    pub fn f64(self) -> Vector<f64, U> {
        Vector {
            e1: self.e1 as f64,
            e2: self.e2 as f64,
            e3: self.e3 as f64,
            marker: PhantomData,
        }
    }
}

impl<T, ULhs, URhs, UOut> ScalarProduct<Vector<T, URhs>> for Vector<T, ULhs>
where
    T: Mul<Output = T> + Add<Output = T> + Copy,
    ULhs: Mul<URhs, Output = UOut>,
{
    type Output = Scalar<T, UOut>;

    fn sp(self, rhs: Vector<T, URhs>) -> Self::Output {
        Scalar {
            value: self.e1 * rhs.e1 + self.e2 * rhs.e2 + self.e3 * rhs.e3,
            marker: PhantomData,
        }
    }
}

impl<T, ULhs, URhs, UOut> BivectorProduct<Vector<T, URhs>> for Vector<T, ULhs>
where
    T: Mul<Output = T> + Sub<Output = T> + Copy,
    ULhs: Mul<URhs, Output = UOut>,
{
    type Output = Bivector<T, UOut>;

    fn bp(self, rhs: Vector<T, URhs>) -> Self::Output {
        Bivector {
            e12: self.e1 * rhs.e2 - self.e2 * rhs.e1,
            e23: self.e2 * rhs.e3 - self.e3 * rhs.e2,
            e31: self.e3 * rhs.e1 - self.e1 * rhs.e3,
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Bivector<T, U> {
    pub e12: T,
    pub e23: T,
    pub e31: T,
    marker: PhantomData<U>,
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Trivector<T, U> {
    pub e123: T,
    marker: PhantomData<U>,
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Multivector<G0, G1, G2, G3>(pub G0, pub G1, pub G2, pub G3);

impl<T, ULhs, URhs, UOut> Mul<Vector<T, URhs>> for Vector<T, ULhs>
where
    Vector<T, ULhs>: ScalarProduct<Vector<T, URhs>, Output = Scalar<T, UOut>>
        + BivectorProduct<Vector<T, URhs>, Output = Bivector<T, UOut>>
        + Copy,
    Vector<T, URhs>: Copy,
    ULhs: Mul<URhs, Output = UOut>,
{
    type Output = Multivector<Scalar<T, UOut>, Zero, Bivector<T, UOut>, Zero>;

    fn mul(self, rhs: Vector<T, URhs>) -> Self::Output {
        Multivector(self.sp(rhs), Zero, self.bp(rhs), Zero)
    }
}

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct Length;

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct Force;

#[allow(non_camel_case_types)]
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct kg_m2_per_s2;

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct KgM2PerS2;

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct KgPerM3;

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct SqrtM;

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct CbrtM2;

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct Energy;

impl Mul<Force> for Length {
    type Output = Energy;
    fn mul(self, _: Force) -> Self::Output {
        Energy
    }
}

impl Mul<Length> for Force {
    type Output = Energy;
    fn mul(self, _: Length) -> Self::Output {
        Energy
    }
}

#[test]
fn unit_conversion() {
    let d: Vector<f64, Length> = Vector::new(0., 1., 0.);
    let f: Vector<f64, Force> = Vector::new(1., 0., 0.);

    let forque = d * f;

    fn name_of<T>(_: &T) -> &'static str {
        std::any::type_name::<T>()
    }

    println!("{}", name_of(&forque));
    // panic!("{:#?}", forque);
}

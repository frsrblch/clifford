use crate::Zero;
use std::ops::{Add, Mul};

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Vector {
    pub e1: f64,
    pub e2: f64,
    pub e3: f64,
}

impl Add<Zero> for Vector {
    type Output = Vector;

    fn add(self, _: Zero) -> Self::Output {
        self
    }
}

impl Add<Vector> for Zero {
    type Output = Vector;

    fn add(self, rhs: Vector) -> Self::Output {
        rhs
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Bivector {
    pub e12: f64,
    pub e23: f64,
    pub e13: f64,
}

impl Add<Zero> for Bivector {
    type Output = Bivector;

    fn add(self, _: Zero) -> Self::Output {
        self
    }
}

impl Add<Bivector> for Zero {
    type Output = Bivector;

    fn add(self, rhs: Bivector) -> Self::Output {
        rhs
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, PartialOrd)]
pub struct Trivector {
    pub e123: f64,
}

impl Add<Zero> for Trivector {
    type Output = Trivector;

    fn add(self, _: Zero) -> Self::Output {
        self
    }
}

impl Add<Trivector> for Zero {
    type Output = Trivector;

    fn add(self, rhs: Trivector) -> Self::Output {
        rhs
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Mv<S, V, B, I>(pub S, pub V, pub B, pub I);

impl<SL, VL, BL, TL, SR, VR, BR, TR, SO, TO, SS, VVS, BBS, TT, ST, VBS, BVS, TS>
    Mul<Mv<SR, VR, BR, TR>> for Mv<SL, VL, BL, TL>
where
    SS: Sum4<VVS, BBS, TT, Output = SO>,
    SL: Mul<SR, Output = SS> + Mul<TR, Output = ST> + Copy,
    VL: ScalarProduct<VR, Output = VVS> + TrivectorProduct<BR, Output = VBS> + Copy,
    BL: ScalarProduct<BR, Output = BBS> + TrivectorProduct<VR, Output = BVS> + Copy,
    TL: Mul<SR, Output = TS> + Mul<TR, Output = TT> + Copy,
    ST: Sum4<VBS, BVS, TS, Output = TO>,
    SR: Copy,
    VR: Copy,
    BR: Copy,
    TR: Copy,
{
    type Output = Mv<SO, Zero, Zero, TO>;

    fn mul(self, rhs: Mv<SR, VR, BR, TR>) -> Self::Output {
        // vector: s*v, v*s, v*b, b*v, b*t, t*b
        // bivect: s*b, v*v, v*t, b*s, b*b, t*v

        Mv(
            Sum4::sum4(
                self.0 * rhs.0,
                self.1.scalar_prod(rhs.1),
                self.2.scalar_prod(rhs.2),
                self.3 * rhs.3,
            ),
            Zero,
            Zero,
            Sum4::sum4(
                self.0 * rhs.3,
                self.1.trivector_prod(rhs.2),
                self.2.trivector_prod(rhs.1),
                self.3 * rhs.0,
            ),
        )
    }
}

pub trait Sum4<B, C, D> {
    type Output;
    fn sum4(self, b: B, c: C, d: D) -> Self::Output;
}

impl<T, A, B, C, D, AB, ABC> Sum4<B, C, D> for A
where
    A: Add<B, Output = AB>,
    AB: Add<C, Output = ABC>,
    ABC: Add<D, Output = T>,
{
    type Output = T;
    fn sum4(self, b: B, c: C, d: D) -> Self::Output {
        self + b + c + d
    }
}

pub trait Sum6<B, C, D, E, F> {
    type Output;
    fn sum6(self, b: B, c: C, d: D, e: E, f: F) -> Self::Output;
}

impl<T, A, B, C, D, E, F, AB, ABC, ABCD, ABCDE> Sum6<B, C, D, E, F> for A
where
    A: Add<B, Output = AB>,
    AB: Add<C, Output = ABC>,
    ABC: Add<D, Output = ABCD>,
    ABCD: Add<E, Output = ABCDE>,
    ABCDE: Add<F, Output = T>,
{
    type Output = T;

    fn sum6(self, b: B, c: C, d: D, e: E, f: F) -> Self::Output {
        self + b + c + d + e + f
    }
}

pub trait ScalarProduct<Rhs> {
    type Output;
    fn scalar_prod(self, rhs: Rhs) -> Self::Output;
}

impl<Lhs, Rhs> ScalarProduct<Rhs> for Lhs
where
    Lhs: Mul<Rhs, Output = f64>,
{
    type Output = f64;

    fn scalar_prod(self, rhs: Rhs) -> Self::Output {
        self * rhs
    }
}

pub trait VectorProduct<Rhs> {
    type Output;
    fn vector_prod(self, rhs: Rhs) -> Self::Output;
}

impl<Lhs, Rhs> VectorProduct<Rhs> for Lhs
where
    Lhs: Mul<Rhs, Output = Vector>,
{
    type Output = Vector;

    fn vector_prod(self, rhs: Rhs) -> Self::Output {
        self * rhs
    }
}

pub trait BivectorProduct<Rhs> {
    type Output;
    fn bivector_prod(self, rhs: Rhs) -> Self::Output;
}

impl<Lhs, Rhs> BivectorProduct<Rhs> for Lhs
where
    Lhs: Mul<Rhs, Output = Bivector>,
{
    type Output = Bivector;

    fn bivector_prod(self, rhs: Rhs) -> Self::Output {
        self * rhs
    }
}

pub trait TrivectorProduct<Rhs> {
    type Output;
    fn trivector_prod(self, rhs: Rhs) -> Self::Output;
}

impl<Lhs, Rhs> TrivectorProduct<Rhs> for Lhs
where
    Lhs: Mul<Rhs, Output = Trivector>,
{
    type Output = Trivector;

    fn trivector_prod(self, rhs: Rhs) -> Self::Output {
        self * rhs
    }
}

impl ScalarProduct<Vector> for Vector {
    type Output = f64;

    fn scalar_prod(self, rhs: Vector) -> Self::Output {
        self.e1 * rhs.e1 + self.e2 * rhs.e2 + self.e3 * rhs.e3
    }
}

impl BivectorProduct<Vector> for Vector {
    type Output = Bivector;

    fn bivector_prod(self, rhs: Vector) -> Self::Output {
        Bivector {
            e12: self.e1 * rhs.e2 - self.e2 * rhs.e1,
            e23: self.e2 * rhs.e3 - self.e3 * rhs.e2,
            e13: self.e1 * rhs.e3 - self.e3 * rhs.e1,
        }
    }
}

impl ScalarProduct<Bivector> for Bivector {
    type Output = f64;

    fn scalar_prod(self, rhs: Bivector) -> Self::Output {
        -(self.e12 * rhs.e12 + self.e13 * rhs.e13 + self.e13 * rhs.e13)
    }
}

impl BivectorProduct<Bivector> for Bivector {
    type Output = Bivector;

    fn bivector_prod(self, rhs: Bivector) -> Self::Output {
        Bivector {
            e12: self.e13 * rhs.e23 - self.e23 * rhs.e12,
            e23: self.e13 * rhs.e12 - self.e12 * rhs.e13,
            e13: self.e13 * rhs.e12 - self.e12 * rhs.e13,
        }
    }
}

#![allow(non_snake_case)]

pub use geo_traits::*;
pub use num_traits::*;

macros::algebra! {
    x ^ 2 == 1,
    y ^ 2 == 1,
    z ^ 2 == 1,
    e ^ 2 == 1,
    E ^ 2 == -1,
}

impl PartialEq<f32> for Scalar<f32> {
    fn eq(&self, other: &f32) -> bool {
        self.s.eq(other)
    }
}

impl PartialEq<f64> for Scalar<f64> {
    fn eq(&self, other: &f64) -> bool {
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

/// Point at the origin
#[inline]
pub fn origin<T: Float>() -> Vector<T> {
    let half = T::one() / (T::one() + T::one());
    Vector {
        e: half,
        E: half,
        ..zero()
    }
}

/// Point through infinity
#[inline]
pub fn infinity<T: Float>() -> Vector<T> {
    Vector {
        e: -T::one(),
        E: T::one(),
        ..zero()
    }
}

#[inline]
pub fn point<T: Float>(x: T, y: T, z: T) -> Vector<T> {
    let half = T::one() / (T::one() + T::one());
    let x2 = x * x + y * y + z * z;
    Vector {
        x,
        y,
        z,
        e: half - half * x2,
        E: half + half * x2,
    }
}

macro_rules! impl_float_type {
    ($($kvector:ident),*) => {
        $(
            impl<T> FloatType for $kvector<T>
            where
                T: num_traits::Float,
            {
                type Float = T;
            }
        )*
    };
}

impl_float_type!(
    Scalar,
    Vector,
    Bivector,
    Trivector,
    Quadvector,
    Pentavector,
    Motor,
    Flector,
    Multivector
);

pub trait IsFlat {
    fn is_flat(&self) -> bool;
}

impl<T, U> IsFlat for T
where
    T: FloatType + Wedge<Vector<T::Float>, Output = U> + Copy,
    U: Zero,
{
    fn is_flat(&self) -> bool {
        self.wedge(infinity::<T::Float>()).is_zero()
    }
}

impl<T: num_traits::Float> Vector<T> {
    pub fn null_bases(self) -> (T, T) {
        let half = T::from(0.5).unwrap();
        ((self.e + self.E), half * (-self.e + self.E))
    }
}

impl<T: num_traits::Float> Vector<T> {
    pub fn vector_type(self) -> VectorType<T> {
        let two = T::one() + T::one();
        let half = two.recip();

        let (n, n_bar) = self.null_bases();
        let Vector { x, y, z, .. } = self;
        let norm2 = x * x + y * y + z * z;

        if n.is_zero() && norm2.is_zero() {
            return VectorType::Infinity;
        }

        if n.is_zero() {
            return VectorType::DualPlane;
        }

        let x2 = norm2 / (n * n);
        let n_bar = n_bar / n;
        let r2 = two * (x2 * half - n_bar);
        let sign = r2.signum();
        VectorType::DualSphere(r2.abs().sqrt().mul(sign))
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum VectorType<T> {
    Infinity,
    DualPlane,
    DualSphere(T),
}

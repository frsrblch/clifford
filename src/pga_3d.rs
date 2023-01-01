pub use geo_traits::*;
pub use num_sqrt::Sqrt;
pub use num_traits::{One, Zero};
pub use num_trig::*;

macros::pga3!();

#[cfg(feature = "dyn")]
macros::dyn_pga3!();

pub fn point<T>(x: T, y: T, z: T) -> Unit<Trivector<T>>
where
    T: One + std::ops::Neg<Output = T>,
{
    Unit::assert(Trivector {
        xyz: T::one(),
        xyw: -z,
        xzw: y,
        yzw: -x,
    })
}

impl<T> num_sqrt::Sqrt for Bivector<T>
where
    Unit<Bivector<T>>: num_sqrt::Sqrt<Output = Motor<T>>,
    Bivector<T>: Unitize<Output = Unit<Bivector<T>>>,
{
    type Output = Motor<T>;
    #[inline]
    fn sqrt(self) -> Self::Output {
        self.unit().sqrt()
    }
}

impl<T> num_sqrt::Sqrt for Unit<Bivector<T>>
where
    Scalar<T>: num_traits::One,
    Bivector<T>: std::ops::Add<Scalar<T>, Output = Motor<T>>,
{
    type Output = Motor<T>;
    #[inline]
    fn sqrt(self) -> Self::Output {
        self.value() + num_traits::One::one()
    }
}

impl<T> num_sqrt::Sqrt for Motor<T>
where
    Motor<T>: Unitize<Output = Unit<Motor<T>>>,
    Unit<Motor<T>>: num_sqrt::Sqrt<Output = Motor<T>>,
{
    type Output = Motor<T>;
    #[inline]
    #[track_caller]
    fn sqrt(self) -> Self::Output {
        self.unit().sqrt()
    }
}

/// Adapted from https://arxiv.org/pdf/2206.07496.pdf, page 14
impl<T> num_sqrt::Sqrt for Unit<Motor<T>>
where
    T: Copy
        + Zero
        + One
        + std::ops::Neg<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + Sqrt<Output = T>,
{
    type Output = Motor<T>;
    #[inline]
    #[track_caller]
    fn sqrt(self) -> Self::Output {
        let one = if self.0.scalar().is_zero() {
            Scalar::<T>::one()
        } else {
            // need to subtract 1 if scalar is negative
            self.0.scalar() / self.0.scalar().norm()
        };
        let x = self.0 + one;
        let s = Scalar::product(x, x.rev());
        let t = Quadvector::product(x, x.rev()).left_comp();
        let a = one / s.sqrt();
        let b = t / (a * a * a);
        a * x + b * Quadvector { xyzw: T::one() } * x
    }
}

impl<T> rand::distributions::Distribution<Unit<Motor<T>>> for rand::distributions::Standard
where
    rand::distributions::Standard:
        rand::distributions::Distribution<Unit<Bivector<T>>> + rand::distributions::Distribution<T>,
    T: std::ops::Mul<Output = T>
        + num_traits::FloatConst
        + num_trig::Sin<Output = T>
        + num_trig::Cos<Output = T>
        + Copy,
    Bivector<T>: std::ops::Mul<Scalar<T>, Output = Bivector<T>>
        + std::ops::Add<Scalar<T>, Output = Motor<T>>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Unit<Motor<T>> {
        let bivector = rng.gen::<Unit<Bivector<T>>>();
        let angle = rng.gen::<T>() * T::PI();
        let sin = Scalar { s: angle.sin() };
        let cos = Scalar { s: angle.cos() };
        Unit(bivector.value() * cos + sin)
    }
}

impl<T> Unit<Trivector<T>>
where
    T: num_traits::Float,
{
    #[inline]
    pub fn x(self) -> T {
        -self.value().yzw
    }

    #[inline]
    pub fn y(self) -> T {
        self.value().xzw
    }

    #[inline]
    pub fn z(self) -> T {
        -self.value().xyw
    }
}

impl<T> Trivector<T>
where
    T: num_traits::Float,
{
    #[inline]
    pub fn x(self) -> T {
        -self.yzw / self.xyz
    }

    #[inline]
    pub fn y(self) -> T {
        self.xzw / self.xyz
    }

    #[inline]
    pub fn z(self) -> T {
        -self.xyw / self.xyz
    }
}

impl<T> Bivector<T>
where
    T: num_traits::Float,
{
    /// Source: Polar decomposition, normalization, square roots and the exponential map in Clifford algebras of fewer than 6 dimensions by Steven De Keninck
    #[inline]
    #[track_caller]
    pub fn exp(self) -> Motor<T> {
        let Bivector {
            xw,
            yw,
            zw,
            yz,
            xz,
            xy,
        } = self;

        let l = yz * yz + xz * xz + xy * xy;

        if l.is_zero() {
            Motor {
                s: T::one(),
                xw,
                yw,
                zw,
                ..Motor::zero()
            }
        } else {
            let m = xw * xy + yw * xz + zw * yz;

            let a = l.sqrt();
            let c = a.cos();
            let s = a.sin() / a;
            let t = m / l * (c - s);
            Motor {
                s: c,
                xw: s * xw + t * xy,
                yw: s * yw + t * xz,
                zw: s * zw + t * yz,
                yz: s * yz,
                xz: s * xz,
                xy: s * xy,
                xyzw: m * s,
            }
        }
    }
}

impl<T> Motor<T>
where
    T: num_traits::Float,
    Motor<T>: Unitize<Output = Unit<Motor<T>>>,
{
    #[inline]
    #[track_caller]
    pub fn log(self) -> Bivector<T> {
        self.unit().log()
    }
}

impl<T: num_traits::Float> Unit<Motor<T>> {
    /// Source: Polar decomposition, normalization, square roots and the exponential map in Clifford algebras of fewer than 6 dimensions by Steven De Keninck
    #[inline]
    #[track_caller]
    pub fn log(mut self) -> Bivector<T> {
        if self.0.s.is_sign_negative() {
            self = Unit::assert(-self.0);
        }

        if self.0.s.is_one() {
            return Bivector {
                xw: self.0.xw,
                yw: self.0.yw,
                zw: self.0.zw,
                ..Bivector::zero()
            };
        }

        let Motor {
            s,
            xw,
            yw,
            zw,
            yz,
            xz,
            xy,
            xyzw,
        } = self.0;

        let a = T::one() / (T::one() - s * s); // inv squared length
        let b = s.acos() * a.sqrt(); // rotation scale
        let c = a * xyzw * (T::one() - s * b); //translation scale

        Bivector {
            xw: c * yz + b * xw,
            yw: c * xz + b * yw,
            zw: c * xy + b * zw,
            yz: b * yz,
            xz: b * xz,
            xy: b * xy,
        }
    }
}

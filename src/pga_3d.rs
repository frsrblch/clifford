pub use geo_traits::*;
pub use num_traits::*;

macros::algebra! {
    x ^ 2 == 1,
    y ^ 2 == 1,
    z ^ 2 == 1,
    w ^ 2 == 0,
}

#[inline]
pub fn point<T>(x: T, y: T, z: T) -> Unit<Trivector<T>>
where
    T: Float,
{
    Unit::assert(Trivector {
        xyz: T::one(),
        xyw: -z,
        xzw: y,
        yzw: -x,
    })
}

impl<T> Bivector<T>
where
    T: Float,
{
    #[inline]
    pub fn sqrt(self) -> Motor<T> {
        self.unit().sqrt()
    }
}

impl<T> Unit<Bivector<T>>
where
    T: Float,
{
    #[inline]
    pub fn sqrt(self) -> Motor<T> {
        self.value() + Scalar::<T>::one()
    }
}

impl<T> Motor<T>
where
    T: Float,
{
    #[inline]
    #[track_caller]
    pub fn sqrt(self) -> Motor<T> {
        self.unit().sqrt()
    }
}

/// Adapted from <https://arxiv.org/pdf/2206.07496.pdf>, page 14
impl<T> Unit<Motor<T>>
where
    T: Float,
{
    #[inline]
    #[track_caller]
    pub fn sqrt(self) -> Motor<T> {
        let one = if self.0.scalar().is_zero() {
            Scalar::<T>::one()
        } else {
            // need to subtract 1 if scalar is negative
            self.0.scalar() / self.0.scalar().norm()
        };
        let x = self.0 + one;
        let t = Quadvector::product(x, x.rev()).left_comp();
        let a = one / x.norm();
        let b = t / (a * a * a);
        a * x + b * Quadvector { xyzw: T::one() } * x
    }
}

impl<T> rand::distributions::Distribution<Unit<Motor<T>>> for rand::distributions::Standard
where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
    T: Float + FloatConst,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Unit<Motor<T>> {
        let plane = rng.gen::<Unit<Bivector<T>>>();
        let angle = rng.gen::<Scalar<T>>() * Scalar::PI();
        let (sin, cos) = angle.sin_cos();
        Unit::assert(plane * cos + sin)
    }
}

impl<T> Trivector<T>
where
    T: Float,
{
    /// Return the (x, y, z) coordinates
    #[inline]
    pub fn coords(self) -> (T, T, T) {
        (
            -self.yzw / self.xyz,
            self.xzw / self.xyz,
            -self.xyw / self.xyz,
        )
    }
}

impl<T> Unit<Trivector<T>>
where
    T: Float,
{
    /// Return the (x, y, z) coordinates
    #[inline]
    pub fn coords(self) -> (T, T, T) {
        (-self.0.yzw, self.0.xzw, -self.0.xyw)
    }
}

impl<T> Bivector<T>
where
    T: Float,
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

        let l = self.norm2().s;

        if l.is_zero() {
            Motor {
                s: T::one(),
                xw,
                yw,
                zw,
                ..zero()
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
    T: Float,
{
    #[inline]
    #[track_caller]
    pub fn log(self) -> Bivector<T> {
        self.unit().log()
    }
}

impl<T> Unit<Motor<T>>
where
    T: num_traits::Float,
{
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
                ..zero()
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

pub use geo_traits::*;
pub use num_sqrt::Sqrt;
pub use num_trig::*;

macros::ga3!();

#[cfg(feature = "dyn")]
macros::dyn_ga3!();

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
    fn sqrt(self) -> Self::Output {
        self.unit().sqrt()
    }
}

impl<T> num_sqrt::Sqrt for Unit<Motor<T>>
where
    Scalar<T>: num_traits::One,
    Motor<T>: std::ops::Add<Scalar<T>, Output = Motor<T>>,
{
    type Output = Motor<T>;
    #[inline]
    fn sqrt(self) -> Self::Output {
        self.value() + num_traits::One::one()
    }
}

impl<T> rand::distributions::Distribution<Unit<Motor<T>>> for rand::distributions::Standard
where
    rand::distributions::Standard:
        rand::distributions::Distribution<Unit<Bivector<T>>> + rand::distributions::Distribution<T>,
    T: std::ops::Mul<Output = T>
        + num_traits::FloatConst
        + num_trig::Trig
        + Copy,
    Bivector<T>: std::ops::Mul<Scalar<T>, Output = Bivector<T>>
        + std::ops::Add<Scalar<T>, Output = Motor<T>>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Unit<Motor<T>> {
        let bivector = rng.gen::<Unit<Bivector<T>>>();
        let angle = Scalar { s: rng.gen::<T>() * T::PI() };
        let (sin, cos) = angle.sin_cos();
        Unit(bivector.value() * cos + sin)
    }
}

impl<T> Motor<T>
where
    T: num_traits::One + std::ops::Add<Output = T> + std::ops::Div<Output = T>,
    Scalar<T>:
        num_trig::Trig + std::ops::Mul<Output = Scalar<T>> + std::ops::Neg<Output = Scalar<T>>,
    Vector<T>: geo_traits::Dual<Output = Bivector<T>>,
    Bivector<T>: std::ops::Mul<Scalar<T>, Output = Bivector<T>>
        + std::ops::Add<Scalar<T>, Output = Motor<T>>,
{
    #[inline]
    pub fn from_axis_and_angle(axis: Unit<Vector<T>>, angle: Scalar<T>) -> Motor<T> {
        let plane = Unit::assert(axis.value().dual());
        Self::from_plane_and_angle(plane, angle)
    }
}

impl<T> Motor<T>
where
    T: num_traits::One + std::ops::Add<Output = T> + std::ops::Div<Output = T>,
    Scalar<T>:
        num_trig::Trig + std::ops::Mul<Output = Scalar<T>> + std::ops::Neg<Output = Scalar<T>>,
    Bivector<T>: std::ops::Mul<Scalar<T>, Output = Bivector<T>>
        + std::ops::Add<Scalar<T>, Output = Motor<T>>,
{
    #[inline]
    pub fn from_plane_and_angle(plane: Unit<Bivector<T>>, angle: Scalar<T>) -> Motor<T> {
        let neg_half = -Scalar {
            s: num_traits::one::<T>() / (num_traits::one::<T>() + num_traits::one::<T>()),
        };
        let (sin, cos) = num_trig::Trig::sin_cos(angle * neg_half);
        plane.value() * sin + cos
    }
}

impl<T> Motor<T>
where
    Bivector<T>: Unitize<Output = Unit<Bivector<T>>> + std::ops::Neg<Output = Bivector<T>>,
{
    #[inline]
    pub fn log(self) -> Unit<Bivector<T>> {
        std::ops::Neg::neg(self.bivector()).unit()
    }
}

impl<T> Motor<T>
where
    Scalar<T>: num_trig::Arccos<Output = Scalar<T>> + std::ops::Mul<Output = Scalar<T>>,
    T: num_traits::One + std::ops::Add<Output = T>,
{
    #[inline]
    pub fn angle(self) -> Scalar<T> {
        let two = num_traits::one::<T>() + num_traits::one::<T>();
        num_trig::Arccos::acos(self.scalar()) * Scalar { s: two }
    }
}

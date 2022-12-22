use geo_traits::*;

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
    T: From<f64>
        + std::ops::Mul<Output = T>
        + num_trig::Sin<Output = T>
        + num_trig::Cos<Output = T>
        + Copy,
    Bivector<T>: std::ops::Mul<Scalar<T>, Output = Bivector<T>>
        + std::ops::Add<Scalar<T>, Output = Motor<T>>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Unit<Motor<T>> {
        let bivector = rng.gen::<Unit<Bivector<T>>>();
        let angle = rng.gen::<T>() * T::from(std::f64::consts::PI);
        let sin = Scalar { s: angle.sin() };
        let cos = Scalar { s: angle.cos() };
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
        let neg_half = -Scalar {
            s: num_traits::one::<T>() / (num_traits::one::<T>() + num_traits::one::<T>()),
        };
        let (sin, cos) = num_trig::Trig::sin_cos(angle * neg_half);
        axis.value().dual() * sin + cos
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

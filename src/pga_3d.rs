use geo_traits::*;

macros::pga3!();

#[cfg(feature = "dyn")]
macros::dyn_pga3!();

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

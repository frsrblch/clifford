use super::*;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, EnumIter, Hash)]
pub enum UnaryTrait {
    /// Marker
    Copy,
    /// std::ops
    Neg,
    Not,
    /// std::iter
    Sum,
    Product,
    /// geo_traits
    Dual,
    LeftComp,
    RightComp,
    Reverse,
    GradeInvolution,
    CliffordConjugate,
    Inverse,
    Unitize,
    Norm2,
    Norm,
    Antinorm2,
    Antinorm,
    Sqrt,
    FloatType,
    Zero,
    One,
    ZeroConst,
    OneConst,
    /// Bytemuck,
    Pod,
    Zeroable,
    /// Rand
    Rand,
}

impl ToTokens for UnaryTrait {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ty().to_tokens(tokens);
    }
}

impl UnaryTrait {
    pub fn ty(&self) -> TokenStream {
        use UnaryTrait::*;
        match self {
            Copy => quote!(std::marker::Copy),
            Neg => quote!(std::ops::Neg),
            Not => quote!(std::ops::Not),
            Sum => quote!(std::iter::Sum),
            Product => quote!(std::iter::Product),
            Dual => quote!(clifford::Dual),
            LeftComp => quote!(clifford::LeftComplement),
            RightComp => quote!(clifford::RightComplement),
            Unitize => quote!(clifford::Unitize),
            Inverse => quote!(clifford::Inv),
            Reverse => quote!(clifford::Reverse),
            GradeInvolution => quote!(clifford::GradeInvolution),
            CliffordConjugate => quote!(clifford::CliffordConjugate),
            Zero => quote!(clifford::Zero),
            One => quote!(clifford::One),
            ZeroConst => quote!(clifford::ZeroConst),
            OneConst => quote!(clifford::OneConst),
            Sqrt => quote!(clifford::Sqrt),
            Norm2 => quote!(clifford::Norm2),
            Norm => quote!(clifford::Norm),
            Antinorm2 => quote!(clifford::Antinorm2),
            Antinorm => quote!(clifford::Antinorm),
            Pod => quote!(clifford::Pod),
            Zeroable => quote!(clifford::Zeroable),
            FloatType => quote!(clifford::FloatType),
            Rand => quote!(rand::distribution::Distribution),
        }
    }

    pub fn fn_ident(&self) -> TokenStream {
        use UnaryTrait::*;
        match self {
            Copy => unimplemented!("Copy fn"),
            Neg => quote!(neg),
            Not => quote!(not),
            Sum => quote!(sum),
            Product => quote!(product),
            Dual => quote!(dual),
            LeftComp => quote!(left_comp),
            RightComp => quote!(right_comp),
            Unitize => quote!(unit),
            Inverse => quote!(inv),
            Reverse => quote!(rev),
            GradeInvolution => quote!(grade_involution),
            CliffordConjugate => quote!(conjugate),
            Zero => quote!(zero),
            One => quote!(one),
            ZeroConst => unimplemented!("ZeroConst fn"),
            OneConst => unimplemented!("OneConst fn"),
            Sqrt => quote!(sqrt),
            Norm2 => quote!(norm2),
            Norm => quote!(norm),
            Antinorm2 => quote!(antinorm2),
            Antinorm => quote!(antinorm),
            Zeroable => quote!(zeroed),
            FloatType | Pod => unimplemented!("FloatType | Pod"),
            Rand => quote!(sample),
        }
    }

    pub fn ty_fn(self) -> (TokenStream, TokenStream) {
        (self.ty(), self.fn_ident())
    }

    pub fn define(self, ty: OverType, algebra: &Algebra) -> Option<Impl<TokenStream>> {
        use FloatParam::T;
        use MagParam::{A, B};
        use UnaryTrait::*;

        if algebra.lean && matches!(self, Not | Sum | Product | Rand) {
            return None;
        }

        match self {
            Sqrt => match ty {
                OverType::Type(Type::Grade(0 | 2))
                | OverType::Type(Type::Motor)
                | OverType::Float(_) => Some(Impl::External),
                _ => None,
            },
            Neg | Reverse | GradeInvolution | CliffordConjugate => match ty {
                OverType::Type(ty) => {
                    let mut bounds = TraitBounds::default();
                    bounds.insert(T);
                    bounds.insert(A);
                    let (trait_ty, trait_fn) = self.ty_fn();
                    let f = match self {
                        Neg => std::ops::Neg::neg,
                        Reverse => Blade::rev,
                        GradeInvolution => Blade::grade_involution,
                        CliffordConjugate => Blade::clifford_conjugate,
                        _ => unimplemented!("Neg | Reverse | GradeInvolution | CliffordConjugate"),
                    };
                    let constructor = Constructor::unary(
                        algebra,
                        &mut bounds,
                        algebra.type_fields(ty),
                        |(blade, field)| ConstructorItem::new(f(blade), quote!(self.#field)),
                    )?;
                    let ty_t = constructor.ty().with_type_param(T, A);
                    let constructor = constructor.into_tokens();
                    let (params, where_clause) = bounds.params_and_where_clause();
                    Some(Impl::Actual(quote! {
                        impl #params #trait_ty for #ty_t #where_clause {
                            type Output = #ty_t;
                            #[inline]
                            fn #trait_fn(self) -> Self::Output {
                                #constructor
                            }
                        }
                    }))
                }
                OverType::Float(_) => match self {
                    Neg => Some(Impl::External),
                    _ => None,
                },
            },
            Dual | LeftComp | RightComp | Not => {
                match (self, algebra.symmetric_complements()) {
                    (Dual | Not, false) => return None,
                    (LeftComp | RightComp, true) => return None,
                    _ => {}
                };
                match ty {
                    OverType::Type(ty) => {
                        let mut bounds = TraitBounds::default();
                        bounds.insert(T);
                        bounds.insert(A);
                        let (trait_ty, trait_fn) = self.ty_fn();

                        let f = {
                            let mut get_output: Box<dyn FnMut(Blade) -> Blade> = match self {
                                Dual | LeftComp | Not => Box::new(|blade| algebra.left_comp(blade)),
                                RightComp => Box::new(|blade| algebra.right_comp(blade)),
                                _ => unreachable!(),
                            };
                            move |blade| {
                                let output = get_output(blade);
                                let field = &algebra.fields[blade];
                                ConstructorItem::new(output, quote!(self.#field))
                            }
                        };

                        let blades = algebra.type_blades(ty);
                        let constructor = Constructor::unary(algebra, &mut bounds, blades, f)?;
                        let mag = if algebra.all_bases_positive() {
                            A
                        } else {
                            MagParam::Mag(Mag::Any)
                        };
                        let output_ty = constructor.ty().with_type_param(T, mag);
                        let constructor = constructor.into_tokens();
                        let (params, where_clause) = bounds.params_and_where_clause();
                        Some(Impl::Actual(quote! {
                            impl #params #trait_ty for #ty<T, A> #where_clause {
                                type Output = #output_ty;
                                #[inline]
                                fn #trait_fn(self) -> Self::Output {
                                    #constructor
                                }
                            }
                        }))
                    }
                    OverType::Float(_) => None,
                }
            }
            Sum => {
                if ty.is_float() {
                    return Some(Impl::External);
                }

                let (trait_ty, trait_fn) = self.ty_fn();
                let (zero_ty, zero_fn) = Zero.ty_fn();
                let mut bounds = TraitBounds::default();
                let ty_t: ParameterizedType = ty.with_type_param(T, A);
                let ty_t_out = ty.with_type_param(T, B);
                bounds.insert(ty_t_out.zero());
                bounds.insert(ty_t_out.add(ty_t, ty_t_out));
                let zero_expr = quote! { <#ty_t_out as #zero_ty>::#zero_fn() };

                let (params, where_clause) = bounds.clone().params_and_where_clause();

                bounds.insert(Lifetime::A);
                bounds.insert(ty_t.copy());
                let (params1, where_clause1) = bounds.params_and_where_clause();

                Some(Impl::Actual(quote! {
                    impl #params #trait_ty<#ty_t> for #ty_t_out #where_clause {
                        fn #trait_fn<I>(iter: I) -> Self where I: Iterator<Item = #ty_t> {
                            iter.fold(#zero_expr, |sum, item| {
                                sum + item
                            })
                        }
                    }
                    impl #params1 #trait_ty<&'a #ty_t> for #ty_t_out #where_clause1 {
                        fn #trait_fn<I>(iter: I) -> Self where I: Iterator<Item = &'a #ty_t> {
                            iter.fold(#zero_expr, |sum, item| {
                                sum + *item
                            })
                        }
                    }
                }))
            }
            Product => {
                if ty.is_float() {
                    return Some(Impl::External);
                }

                let inner = Type::from(ty);
                if algebra.product(ty, ty, Algebra::geo) != Some(inner) || !inner.contains(Blade(0))
                {
                    return None;
                }

                let (product_ty, product_fn) = Product.ty_fn();
                let (one_ty, one_fn) = One.ty_fn();
                let mut bounds = TraitBounds::default();
                let ty_t = ty.with_type_param(T, A);

                bounds.insert(ty_t.one());
                let one_expr = quote!(<#ty_t as #one_ty>::#one_fn());

                bounds.insert(A);
                bounds.insert(ty_t.mul(ty_t, ty_t));
                let (params, where_clause) = bounds.clone().params_and_where_clause();

                bounds.insert(Lifetime::A);
                bounds.insert(ty_t.copy());
                let (params1, where_clause1) = bounds.params_and_where_clause();

                Some(Impl::Actual(quote! {
                    impl #params #product_ty<#ty_t> for #ty_t #where_clause {
                        fn #product_fn<I>(iter: I) -> Self where I: Iterator<Item = #ty_t> {
                            iter.fold(#one_expr, |product, item| {
                                product * item
                            })
                        }
                    }
                    impl #params1 #product_ty<&'a #ty_t> for #ty_t #where_clause1 {
                        fn #product_fn<I>(iter: I) -> Self where I: Iterator<Item = &'a #ty_t> {
                            iter.fold(#one_expr, |product, item| {
                                product * *item
                            })
                        }
                    }
                }))
            }
            Zero => match ty {
                OverType::Type(ty) => {
                    // Zero requires Add<Self, Output = Self>
                    if BinaryTrait::Add.no_impl(OverType::Type(ty), OverType::Type(ty), algebra) {
                        return None;
                    }

                    let (zero_ty, zero_fn) = Zero.ty_fn();

                    let mut bounds = TraitBounds::default();
                    let ty_t = ty.with_type_param(T, A);
                    bounds.insert(ty_t.add(ty_t, ty_t));
                    bounds.insert(T.partial_eq(T));
                    bounds.insert(T.zero());

                    let constructor = Constructor::unary(
                        algebra,
                        &mut bounds,
                        algebra.type_blades(ty),
                        |blade| ConstructorItem::new(blade, quote!(#zero_ty::#zero_fn())),
                    )?
                    .into_tokens();

                    let (params, where_clause) = bounds.params_and_where_clause();

                    Some(Impl::Actual(quote! {
                        impl #params #zero_ty for #ty_t #where_clause {
                            #[inline]
                            fn #zero_fn() -> Self {
                                #constructor
                            }
                            #[inline]
                            fn is_zero(&self) -> bool {
                                <Self as #zero_ty>::#zero_fn().eq(self)
                            }
                        }
                    }))
                }
                OverType::Float(_) => Some(Impl::External),
            },
            ZeroConst => match ty {
                OverType::Type(ty) => {
                    // Zero requires Add<Self, Output = Self>
                    if BinaryTrait::Add.no_impl(OverType::Type(ty), OverType::Type(ty), algebra) {
                        return None;
                    }
                    let mut bounds = TraitBounds::default();
                    bounds.insert(T.zero_const());
                    let zero_const_ty = self.ty();

                    let constructor = Constructor::new(
                        algebra,
                        &mut bounds,
                        algebra.type_blades(ty),
                        |blade, _| ConstructorItem::new(blade, quote!(T::ZERO)),
                    )
                    .map(Constructor::into_tokens)?;

                    let ty_t = ty.with_type_param(T, Mag::Any);
                    let (params, where_clause) = bounds.params_and_where_clause();

                    Some(Impl::Actual(quote! {
                        impl #params #zero_const_ty for #ty_t #where_clause {
                            const ZERO: #ty_t = #constructor;
                        }
                    }))
                }
                OverType::Float(_) => Some(Impl::External),
            },
            One => match ty {
                OverType::Type(ty) => {
                    // One requires Mul<Self, Output = Self>
                    if BinaryTrait::Mul.no_impl(OverType::Type(ty), OverType::Type(ty), algebra)
                        || !ty.contains_grade(0)
                    {
                        return None;
                    }

                    let (trait_ty, trait_fn) = self.ty_fn();
                    let mut bounds = TraitBounds::default();
                    let ty_t = ty.with_type_param(T, A);
                    bounds.insert(ty_t.mul(ty_t, ty_t));

                    let (one_ty, one_fn) = One.ty_fn();
                    let (zero_ty, zero_fn) = Zero.ty_fn();
                    let constructor = Constructor::new(
                        algebra,
                        &mut bounds,
                        algebra.type_blades(ty),
                        |blade, bounds| {
                            let tokens = if blade.is_scalar() {
                                bounds.insert(T.one());
                                quote!(#one_ty::#one_fn())
                            } else {
                                bounds.insert(T.zero());
                                quote!(#zero_ty::#zero_fn())
                            };
                            ConstructorItem::new(blade, tokens)
                        },
                    )?;

                    let constructor = constructor.into_token_stream();

                    let (params, where_clause) = bounds.params_and_where_clause();

                    Some(Impl::Actual(quote! {
                        impl #params #trait_ty for #ty_t #where_clause {
                            #[inline]
                            fn #trait_fn() -> Self {
                                #constructor
                            }
                        }
                    }))
                }
                OverType::Float(_) => Some(Impl::External),
            },
            OneConst => match ty {
                OverType::Type(ty) => {
                    // One requires Mul<Self, Output = Self>
                    if BinaryTrait::Mul.no_impl(OverType::Type(ty), OverType::Type(ty), algebra) {
                        return None;
                    }

                    let mut bounds = TraitBounds::default();
                    bounds.insert(T);
                    bounds.insert(A);

                    let ty_t = ty.with_type_param(T, A);

                    let geo_traits_one_ty = quote!(clifford::OneConst);
                    let mut zero_bound = false;
                    let const_fields = algebra
                        .type_fields(ty)
                        .map(|(blade, field)| {
                            if blade == Blade::scalar() {
                                bounds.insert(T.one_const());
                                quote!(#field: T::ONE,)
                            } else {
                                zero_bound = true;
                                bounds.insert(T.zero_const());
                                quote!(#field: T::ZERO,)
                            }
                        })
                        .collect::<Vec<_>>();

                    let (params, where_clause) = bounds.params_and_where_clause();

                    Some(Impl::Actual(quote! {
                        impl #params #geo_traits_one_ty for #ty_t #where_clause {
                            const ONE: #ty_t = #ty {
                                #(#const_fields)*
                                marker: std::marker::PhantomData,
                            };
                        }
                    }))
                }
                OverType::Float(_) => Some(Impl::External),
            },
            Norm2 | Antinorm2 => match ty {
                OverType::Type(_) => {
                    let mut bounds = TraitBounds::default();
                    bounds.insert(MagParam::A);
                    bounds.insert(T.copy());

                    let (trait_ty, trait_fn) = self.ty_fn();
                    let ty_t = ty.with_type_param(T, MagParam::A);

                    let f: Box<dyn Fn(Blade) -> Blade> = match self {
                        Norm2 => Box::new(|blade| algebra.dot(blade, blade.rev())),
                        Antinorm2 => {
                            Box::new(|blade| algebra.antidot(blade, algebra.antirev(blade)))
                        }
                        _ => unreachable!(),
                    };

                    bounds.insert(T.mul(T, T));

                    let constructor = Constructor::unary(
                        algebra,
                        &mut bounds,
                        algebra.type_fields(ty),
                        |(blade, field)| {
                            ConstructorItem::new(f(blade), quote! { self.#field * self.#field })
                        },
                    )?;

                    let output_ty = constructor.ty().with_type_param(T, MagParam::A);

                    let constructor = constructor.into_tokens();

                    let (params, where_clause) = bounds.params_and_where_clause();
                    Some(Impl::Actual(quote! {
                        impl #params #trait_ty for #ty_t #where_clause {
                            type Output = #output_ty;
                            #[inline]
                            fn #trait_fn(self) -> Self::Output {
                                #constructor
                            }
                        }
                    }))
                }
                OverType::Float(_) => None,
            },
            Norm | Antinorm => match ty {
                OverType::Type(ty) => {
                    let (trait_ty, trait_fn) = self.ty_fn();
                    let ty_t = ty.with_type_param(T, A);

                    let mut bounds = TraitBounds::default();
                    bounds.insert(ty_t);
                    bounds.insert(T.copy());
                    bounds.insert(T.sqrt());
                    bounds.insert(T.mul(T, T));

                    let mut constructor = Constructor::unary(
                        algebra,
                        &mut bounds,
                        algebra.type_fields(ty),
                        |(blade, field)| {
                            let output = match self {
                                Norm => algebra.dot(blade, blade.rev()),
                                Antinorm => algebra.antidot(blade, algebra.antirev(blade)),
                                _ => unreachable!(),
                            };
                            ConstructorItem::new(output, quote!(self.#field * self.#field))
                        },
                    )?;

                    let (sqrt_ty, sqrt_fn) = Sqrt.ty_fn();
                    constructor
                        .blades
                        .values_mut()
                        .for_each(|ts| *ts = quote!(#sqrt_ty::#sqrt_fn(#ts)));

                    let output_t = constructor.ty().with_type_param(T, A);

                    let constructor = constructor.into_token_stream();

                    let (params, where_clause) = bounds.params_and_where_clause();
                    Some(Impl::Actual(quote! {
                        impl #params #trait_ty for #ty_t #where_clause {
                            type Output = #output_t;
                            #[inline]
                            fn #trait_fn(self) -> Self::Output {
                                #constructor
                            }
                        }
                    }))
                }
                OverType::Float(_) => None,
            },
            Inverse if matches!(ty, OverType::Float(Float::F32 | Float::F64)) => {
                Some(Impl::External)
            }
            Unitize | Inverse => match ty {
                OverType::Type(ty) => {
                    let mut bounds = TraitBounds::default();
                    let ty_t = ty.with_type_param(T, A);

                    bounds.insert(ty_t);
                    bounds.insert(T.copy());

                    let (trait_ty, trait_fn) = self.ty_fn();

                    let norm_var = match self {
                        Unitize => quote!(norm),
                        Inverse => quote!(norm2),
                        _ => unreachable!(),
                    };

                    let output_t = match self {
                        Unitize => ty.with_type_param(T, Mag::Unit),
                        Inverse => ty_t,
                        _ => unreachable!(),
                    };

                    bounds.insert(T.mul(T, T));
                    let sum = Constructor::unary(
                        algebra,
                        &mut bounds,
                        algebra.type_fields(ty),
                        |(blade, field)| {
                            ConstructorItem::new(
                                algebra.dot(blade, blade.rev()),
                                quote!(self.#field * self.#field),
                            )
                        },
                    )?
                    .blades
                    .remove(&Blade(0))?;

                    if sum.is_empty() {
                        return None;
                    }

                    let norm_expr = match self {
                        Unitize => {
                            let (sqrt_ty, sqrt_fn) = Sqrt.ty_fn();
                            bounds.insert(T.sqrt());
                            quote!(#sqrt_ty::#sqrt_fn(#sum))
                        }
                        Inverse => sum,
                        _ => unreachable!(),
                    };
                    bounds.insert(T.div(T, T));

                    let constructor = Constructor::unary(
                        algebra,
                        &mut bounds,
                        algebra.type_fields(ty),
                        |(blade, field)| match self {
                            Unitize => ConstructorItem::new(blade, quote!(self.#field / #norm_var)),
                            Inverse => {
                                ConstructorItem::new(blade.rev(), quote!(self.#field / #norm_var))
                            }
                            _ => unreachable!(),
                        },
                    )
                    .map(Constructor::into_token_stream)?;

                    let (params, where_clause) = bounds.params_and_where_clause();

                    Some(Impl::Actual(quote! {
                        impl #params #trait_ty for #ty_t #where_clause {
                            type Output = #output_t;
                            #[inline]
                            fn #trait_fn(self) -> Self::Output {
                                let #norm_var = #norm_expr;
                                #constructor
                            }
                        }
                    }))
                }
                _ => None,
            },
            Pod => {
                if ty.is_float() {
                    return Some(Impl::External);
                }
                let trait_ty = self.ty();
                let zeroable_ty = Zeroable.ty();
                let ty_t = ty.with_type_param(T, MagParam::A);
                Some(Impl::Actual(quote! {
                    unsafe impl<T, A> #trait_ty for #ty_t
                    where
                        T: #zeroable_ty + #trait_ty,
                        Self: Copy + 'static,
                    {}
                }))
            }
            Zeroable => {
                // while unit types are not valid in a zeroed state, they are still memory safe
                if ty.is_float() {
                    return Some(Impl::External);
                }
                let ty_t = ty.with_type_param(T, MagParam::A);
                let trait_ty = self.ty();
                Some(Impl::Actual(quote! {
                    unsafe impl<T, A> #trait_ty for #ty_t
                    where
                        T: #trait_ty,
                    {}
                }))
            }
            FloatType => {
                if ty.is_float() {
                    return None;
                }
                let trait_ty = self.ty();
                let ty_t = ty.with_type_param(T, A);
                Some(Impl::Actual(quote! {
                    impl<T, A> #trait_ty for #ty_t {
                        type Float = T;
                    }
                }))
            }
            Rand => match ty {
                OverType::Float(_) => Some(Impl::External),
                OverType::Type(Type::Motor) | OverType::Type(Type::Flector) => {
                    let mut bounds = TraitBounds::default();
                    let (t, a) = (T, A);
                    bounds.insert(t);
                    bounds.insert(a);
                    let ty_t = ty.with_type_param(t, a);
                    let vec_t = Type::Grade(1).with_type_param(t, a);
                    let motor_t = Type::Motor.with_type_param(t, a);
                    bounds.insert(vec_t.mul(vec_t, motor_t));
                    if ty == OverType::Type(Type::Flector) {
                        let flector_t = Type::Flector.with_type_param(t, a);
                        bounds.insert(motor_t.mul(vec_t, flector_t));
                    }
                    let expr = match ty {
                        OverType::Type(Type::Motor) => {
                            quote!(rand::Rng::random::<#vec_t>(rng) * rand::Rng::random::<#vec_t>(rng))
                        }
                        OverType::Type(Type::Flector) => {
                            quote!(rand::Rng::random::<#vec_t>(rng) * rand::Rng::random::<#vec_t>(rng) * rand::Rng::random::<#vec_t>(rng))
                        }
                        _ => unreachable!(),
                    };
                    let (params, where_clause) = bounds.params_and_where_clause();

                    Some(Impl::Actual(quote! {
                        impl #params rand::distr::Distribution<#ty_t> for rand::distr::StandardUniform
                            #where_clause
                                rand::distr::StandardUniform: rand::distr::Distribution<#t> + rand::distr::Distribution<#vec_t>,
                        {
                            #[inline]
                            fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) ->  #ty_t {
                                #expr
                            }
                        }
                    }))
                }
                OverType::Type(Type::Grade(_)) => {
                    let inner = Type::from(ty);
                    let t = T;
                    let ty_t = ty.with_type_param(T, Mag::Any);
                    let unit_t = ty.with_type_param(T, Mag::Unit);
                    let (one_ty, one_fn) = One.ty_fn();
                    let (zero_ty, zero_fn) = Zero.ty_fn();
                    let (norm2_ty, norm2_fn) = Norm2.ty_fn();
                    let mut bounds = TraitBounds::default();
                    bounds.insert(t.one());
                    let mut has_non_zero = false;
                    let fields = algebra
                        .type_fields(ty)
                        .map(|(blade, field)| {
                            if algebra.dot(blade, blade).is_zero() {
                                bounds.insert(t.zero());
                                quote!(#field: #zero_ty::#zero_fn())
                            } else {
                                has_non_zero = true;
                                bounds.insert(t.mul(t, t));
                                bounds.insert(t.sub(t, t));
                                quote!(#field: rand::Rng::random_range(rng, -one..=one))
                            }
                        })
                        .collect::<Vec<_>>();

                    if !has_non_zero {
                        return None;
                    }

                    let s = &algebra.fields[Blade(0)];

                    let inner_t = inner.with_type_param(t, Mag::Any);
                    bounds.insert(inner_t.norm2());
                    let (params, where_clause) = bounds.params_and_where_clause();

                    Some(Impl::Actual(quote! {
                        impl #params rand::distr::Distribution<#ty_t> for rand::distr::StandardUniform
                        #where_clause
                            #t: clifford::Number + rand::distr::uniform::SampleUniform,
                            std::ops::RangeInclusive<#t>: rand::distr::uniform::SampleRange<#t>,
                        {
                            #[inline]
                            fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) ->  #ty_t {
                                let one = <#t as #one_ty>::#one_fn();
                                for _ in 0..64 {
                                    let v = #inner {
                                        #(#fields,)*
                                        marker: std::marker::PhantomData,
                                    };
                                    let norm2 = #norm2_ty::#norm2_fn(v).#s;
                                    if norm2 <= one {
                                        return v;
                                    }
                                }
                                panic!("unable to find unit value for {}", std::any::type_name::<Self>());
                            }
                        }
                        impl #params rand::distr::Distribution<#unit_t> for rand::distr::StandardUniform
                        #where_clause
                            #t: clifford::Number + rand::distr::uniform::SampleUniform,
                            std::ops::RangeInclusive<#t>: rand::distr::uniform::SampleRange<#t>,
                        {
                            #[inline]
                            fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) ->  #unit_t {
                                let one = <#t as #one_ty>::#one_fn();
                                for _ in 0..64 {
                                    let v = #inner {
                                        #(#fields,)*
                                        marker: std::marker::PhantomData,
                                    };
                                    let norm2 = #norm2_ty::#norm2_fn(v).#s;
                                    if norm2 <= one {
                                        return (v / Scalar::new(norm2.sqrt())).assert()
                                    }
                                }
                                panic!("unable to find unit value for {}", std::any::type_name::<Self>());
                            }
                        }
                    }))
                }
            },
            Copy => Some(Impl::External),
        }
    }

    pub fn impl_type(&self, ty: OverType, algebra: &Algebra) -> Option<Impl<()>> {
        self.define(ty, algebra).map(|opt| opt.map(|_| ()))
    }

    pub fn no_impl<T>(self, ty: T, algebra: &Algebra) -> bool
    where
        OverType: From<T>,
    {
        self.impl_type(ty.into(), algebra).is_none()
    }

    pub fn define_tests(self, ty: Type, algebra: &Algebra) -> Option<TokenStream> {
        use UnaryTrait::*;
        if self.no_impl(ty, algebra) {
            return None;
        }
        match self {
            Inverse => {
                let product = algebra.product(ty, ty, Algebra::geo)?;
                if BinaryTrait::Sub.no_impl(product, product, algebra)
                    || Norm2.no_impl(product, algebra)
                    || Rand.no_impl(Type::Grade(1), algebra)
                    || algebra.has_negative_bases()
                {
                    return None;
                }
                let ty_ident = ty.fn_ident();
                let fn_ident = format_ident!("inverse_{ty_ident}");
                let ty_t = ty.with_type_param(Float::F64, Mag::Any);
                let (inv_ty, inv_fn) = Inverse.ty_fn();
                let (norm2_ty, norm2_fn) = Norm2.ty_fn();
                let s = &algebra.fields[Blade(0)];
                let vec_t = Type::Grade(1).with_type_param(Float::F64, Mag::Any);
                let value_expr = match ty {
                    Type::Grade(_) => quote!(rng.random::<#ty_t>()),
                    Type::Motor => quote!(rng.random::<#vec_t>() * rng.random::<#vec_t>()),
                    Type::Flector => {
                        quote!(rng.random::<#vec_t>() * rng.random::<#vec_t>() * rng.random::<#vec_t>())
                    }
                };
                Some(quote! {
                    #[test]
                    fn #fn_ident() {
                        use rand::Rng;
                        let mut rng = rand::rng();
                        for _ in 0..100 {
                            let value = #value_expr;
                            let inv = #inv_ty::#inv_fn(value);
                            let product = value * inv;
                            let expected = #product {
                                #s: 1f64,
                                ..clifford::Zero::zero()
                            };
                            let diff = product - expected;
                            assert!(#norm2_ty::#norm2_fn(diff).#s.abs() < 1e-10);
                        }
                    }
                })
            }
            Unitize => {
                let product = algebra.product(ty, ty, Algebra::geo)?;
                if BinaryTrait::Sub.no_impl(product, product, algebra)
                    || Norm2.no_impl(product, algebra)
                    || Rand.no_impl(Type::Grade(1), algebra)
                    || Reverse.no_impl(ty, algebra)
                    || algebra.has_negative_bases()
                {
                    return None;
                }
                let ty_ident = ty.fn_ident();
                let fn_ident = format_ident!("unitize_{ty_ident}");
                let ty_t = ty.with_type_param(Float::F64, Mag::Unit);
                let (norm2_ty, norm2_fn) = Norm2.ty_fn();
                let (rev_ty, rev_fn) = Reverse.ty_fn();
                let (zero_ty, zero_fn) = Zero.ty_fn();
                let s = &algebra.fields[Blade(0)];
                let vec_t_unit = Type::Grade(1).with_type_param(Float::F64, Mag::Unit);
                let unit_value_expr = match ty {
                    Type::Grade(_) => quote!(rng.random::<#ty_t>()),
                    Type::Motor => {
                        quote!(rng.random::<#vec_t_unit>() * rng.random::<#vec_t_unit>())
                    }
                    Type::Flector => {
                        quote!(rng.random::<#vec_t_unit>() * rng.random::<#vec_t_unit>() * rng.random::<#vec_t_unit>())
                    }
                };
                Some(quote! {
                    #[test]
                    fn #fn_ident() {
                        use rand::{Rng, rng};
                        let mut rng = rng();
                        for _ in 0..100 {
                            let unit = #unit_value_expr;
                            let product = unit * #rev_ty::#rev_fn(unit);
                            let expected = #product {
                                #s: 1f64,
                                ..#zero_ty::#zero_fn()
                            };
                            let diff = product - expected;
                            assert!(#norm2_ty::#norm2_fn(diff).#s.abs() < 1e-10);
                        }
                    }
                })
            }
            _ => None,
        }
    }
}

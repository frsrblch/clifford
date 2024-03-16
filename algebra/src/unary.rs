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
    FloatType,
    Zero,
    One,
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
            Norm2 => quote!(clifford::Norm2),
            Norm => quote!(clifford::Norm),
            Pod => quote!(bytemuck::Pod),
            Zeroable => quote!(bytemuck::Zeroable),
            FloatType => quote!(clifford::FloatType),
            Rand => quote!(rand::distribution::Distribution),
        }
    }

    pub fn fn_ident(&self) -> TokenStream {
        use UnaryTrait::*;
        match self {
            Copy => unimplemented!(),
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
            Norm2 => quote!(norm2),
            Norm => quote!(norm),
            Zeroable => quote!(zeroed),
            FloatType | Pod => unimplemented!(),
            Rand => quote!(sample),
        }
    }

    pub fn ty_fn(self) -> (TokenStream, TokenStream) {
        (self.ty(), self.fn_ident())
    }

    pub fn define(self, ty: OverType, algebra: &Algebra) -> Impl<TokenStream> {
        use FloatParam::T;
        use MagParam::A;
        use UnaryTrait::*;

        match self {
            Neg | Reverse | GradeInvolution | CliffordConjugate => match ty {
                OverType::Type(ty) => {
                    let (trait_ty, trait_fn) = self.ty_fn();
                    let f = match self {
                        Neg => std::ops::Neg::neg,
                        Reverse => Blade::rev,
                        GradeInvolution => Blade::grade_involution,
                        CliffordConjugate => Blade::clifford_conjugate,
                        _ => unimplemented!(),
                    };
                    let mut bounds = TraitBounds::default();
                    bounds.insert(T);
                    bounds.insert(A);
                    let ty_t = ty.with_type_param(T, A);
                    let fields = TypeBlades::new(algebra, ty)
                        .map(|blade| {
                            let field = &algebra.fields[blade];
                            if f(blade) == blade {
                                quote!(#field: self.#field,)
                            } else {
                                bounds.insert(T.neg());
                                quote!(#field: -self.#field,)
                            }
                        })
                        .collect::<Vec<_>>();
                    let (params, where_clause) = bounds.params_and_where_clause();
                    Impl::Actual(quote! {
                        impl #params #trait_ty for #ty_t #where_clause {
                            type Output = #ty_t;
                            #[inline]
                            fn #trait_fn(self) -> Self::Output {
                                #ty {
                                    #(#fields)*
                                    marker: std::marker::PhantomData,
                                }
                            }
                        }
                    })
                }
                OverType::Float(_) => match self {
                    Neg => Impl::External,
                    _ => Impl::None,
                },
            },
            Dual | LeftComp | RightComp | Not => {
                match (self, algebra.symmetric_complements()) {
                    (Dual | Not, false) | (LeftComp | RightComp, true) => return Impl::None,
                    _ => {}
                };
                match ty {
                    OverType::Type(ty) => {
                        let mut bounds = TraitBounds::with_param(T);
                        bounds.insert(A);
                        let (trait_ty, trait_fn) = self.ty_fn();
                        let output = ty.complement(algebra);
                        let fields = TypeFields::new(algebra, output)
                            .map(|(blade, field)| {
                                // these are left comp and right comp are reversed
                                // because we're looking at the complement of the
                                let comp = if self == LeftComp {
                                    algebra.right_comp(blade)
                                } else {
                                    algebra.left_comp(blade)
                                };
                                let sign = if comp.is_negative() {
                                    bounds.insert(T.neg());
                                    quote!(-)
                                } else {
                                    quote!()
                                };
                                let comp_field = &algebra.fields[comp];
                                quote! {
                                    #field: #sign self.#comp_field,
                                }
                            })
                            .collect::<Vec<_>>();

                        let output_ty = if algebra.all_bases_positive() {
                            output.with_type_param(T, A)
                        } else {
                            output.with_type_param(T, MagParam::Mag(Mag::Any))
                        };

                        let (params, where_clause) = bounds.params_and_where_clause();
                        Impl::Actual(quote! {
                            impl #params #trait_ty for #ty<T, A> #where_clause {
                                type Output = #output_ty;
                                #[inline]
                                fn #trait_fn(self) -> Self::Output {
                                    #output {
                                        #(#fields)*
                                        marker: std::marker::PhantomData,
                                    }
                                }
                            }
                        })
                    }
                    OverType::Float(_) => Impl::None,
                }
            }
            Sum => {
                if ty.is_float() {
                    return Impl::External;
                }

                let (trait_ty, trait_fn) = self.ty_fn();
                let (zero_ty, zero_fn) = UnaryTrait::Zero.ty_fn();
                let mut bounds = TraitBounds::default();
                let ty_t = ty.with_type_param(T, A);
                let ty_t_any = ty.with_type_param(T, Mag::Any);
                bounds.insert(T);
                bounds.insert(A);
                bounds.insert(ty_t_any.zero());
                bounds.insert(ty_t_any.add(ty_t, ty_t_any));
                let (params, where_clause) = bounds.clone().params_and_where_clause();

                bounds.insert(Lifetime::A);
                bounds.insert(ty_t.copy());
                let (params1, where_clause1) = bounds.params_and_where_clause();

                Impl::Actual(quote! {
                    impl #params #trait_ty<#ty_t> for #ty_t_any #where_clause {
                        fn #trait_fn<I>(iter: I) -> Self where I: Iterator<Item = #ty_t> {
                            iter.fold(<#ty_t_any as #zero_ty>::#zero_fn(), |sum, item| {
                                sum + item
                            })
                        }
                    }
                    impl #params1 #trait_ty<&'a #ty_t> for #ty_t_any #where_clause1 {
                        fn #trait_fn<I>(iter: I) -> Self where I: Iterator<Item = &'a #ty_t> {
                            iter.fold(<#ty_t_any as #zero_ty>::#zero_fn(), |sum, item| {
                                sum + *item
                            })
                        }
                    }
                })
            }
            Product => {
                if ty.is_float() {
                    return Impl::External;
                }

                let inner = Type::from(ty);
                if algebra.product(ty, ty, Algebra::geo) != Some(inner) || !inner.contains(Blade(0))
                {
                    return Impl::None;
                }

                let (trait_ty, trait_fn) = self.ty_fn();
                let (one_ty, one_fn) = UnaryTrait::One.ty_fn();
                let (zero_ty, zero_fn) = UnaryTrait::Zero.ty_fn();
                let mut bounds = TraitBounds::default();
                let ty_t = ty.with_type_param(T, A);
                bounds.insert(T);

                let one_expr = {
                    let fields = TypeFields::new(algebra, ty).map(|(b, f)| {
                        if b == Blade(0) {
                            bounds.insert(T.one());
                            quote! {
                                #f: <T as #one_ty>::#one_fn()
                            }
                        } else {
                            bounds.insert(T.zero());
                            quote! {
                                #f: <T as #zero_ty>::#zero_fn()
                            }
                        }
                    });
                    quote! {
                        #inner::<T, A> {
                            #(#fields,)*
                            marker: std::marker::PhantomData,
                        }
                    }
                };

                bounds.insert(A);
                bounds.insert(ty_t.mul(ty_t, ty_t));
                let (params, where_clause) = bounds.clone().params_and_where_clause();

                bounds.insert(Lifetime::A);
                bounds.insert(ty_t.copy());
                let (params1, where_clause1) = bounds.params_and_where_clause();

                Impl::Actual(quote! {
                    impl #params #trait_ty<#ty_t> for #ty_t #where_clause {
                        fn #trait_fn<I>(iter: I) -> Self where I: Iterator<Item = #ty_t> {
                            iter.fold(#one_expr, |sum, item| {
                                sum * item
                            })
                        }
                    }
                    impl #params1 #trait_ty<&'a #ty_t> for #ty_t #where_clause1 {
                        fn #trait_fn<I>(iter: I) -> Self where I: Iterator<Item = &'a #ty_t> {
                            iter.fold(#one_expr, |sum, item| {
                                sum * *item
                            })
                        }
                    }
                })
            }
            Zero => match ty {
                OverType::Type(ty) => {
                    // Zero requires Add<Self, Output = Self>
                    if BinaryTrait::Add.no_impl(OverType::Type(ty), OverType::Type(ty), algebra) {
                        return Impl::None;
                    }

                    let (trait_ty, trait_fn) = self.ty_fn();
                    let fields = TypeFields::new(algebra, ty).map(|(_, field)| {
                        quote! {
                            #field: #trait_ty::#trait_fn(),
                        }
                    });

                    let geo_traits_zero_ty = quote!(clifford::ZeroConst);
                    let const_fields =
                        TypeFields::new(algebra, ty).map(|(_, field)| quote!(#field: T::ZERO,));
                    let ty_t = ty.with_type_param(T, Mag::Any);
                    Impl::Actual(quote! {
                        impl<T> #geo_traits_zero_ty for #ty_t where T: #geo_traits_zero_ty {
                            const ZERO: #ty_t = #ty {
                                #(#const_fields)*
                                marker: std::marker::PhantomData,
                            };
                        }

                        impl<T> #trait_ty for #ty_t
                        where
                            T: #trait_ty + PartialEq,
                            #ty_t: std::ops::Add<Output = #ty<T, Any>>,
                        {
                            #[inline]
                            fn #trait_fn() -> Self {
                                #ty {
                                    #(#fields)*
                                    marker: std::marker::PhantomData,
                                }
                            }
                            #[inline]
                            fn is_zero(&self) -> bool {
                                <Self as #trait_ty>::#trait_fn().eq(self)
                            }
                        }
                    })
                }
                OverType::Float(_) => Impl::External,
            },
            One => match ty {
                OverType::Type(ty) => {
                    // One requires Mul<Self, Output = Self>
                    if BinaryTrait::Mul.no_impl(OverType::Type(ty), OverType::Type(ty), algebra) {
                        return Impl::None;
                    }

                    let (trait_ty, trait_fn) = self.ty_fn();
                    let (zero_ty, zero_fn) = Self::Zero.ty_fn();
                    let t = T;
                    let mut bounds = TraitBounds::default();
                    bounds.insert(A);
                    bounds.insert(t.one());
                    bounds.insert(t.partial_eq(t));
                    bounds.insert(t.copy());
                    let ty_t = ty.with_type_param(t, A);
                    bounds.insert(ty_t.mul(ty_t, ty_t));
                    let fields = TypeFields::new(algebra, ty)
                        .map(|(blade, field)| {
                            if blade == Blade(0) {
                                quote! {
                                    #field: #trait_ty::#trait_fn(),
                                }
                            } else {
                                bounds.insert(t.zero());
                                quote! {
                                    #field: #zero_ty::#zero_fn(),
                                }
                            }
                        })
                        .collect::<Vec<_>>();

                    let (params, where_clause) = bounds.params_and_where_clause();

                    let geo_traits_one_ty = quote!(clifford::OneConst);
                    let mut zero_bound = false;
                    let const_fields = TypeFields::new(algebra, ty)
                        .map(|(blade, field)| {
                            if blade == Blade::scalar() {
                                quote!(#field: T::ONE,)
                            } else {
                                zero_bound = true;
                                quote!(#field: T::ZERO,)
                            }
                        })
                        .collect::<Vec<_>>();
                    let plus_geo_traits_zero_ty = if zero_bound {
                        quote!(+ clifford::ZeroConst)
                    } else {
                        quote!()
                    };

                    Impl::Actual(quote! {
                        impl #params #geo_traits_one_ty for #ty_t where T: #geo_traits_one_ty #plus_geo_traits_zero_ty {
                            const ONE: #ty_t = #ty {
                                #(#const_fields)*
                                marker: std::marker::PhantomData,
                            };
                        }
                        impl #params #trait_ty for #ty_t #where_clause {
                            #[inline]
                            fn #trait_fn() -> Self {
                                #ty {
                                    #(#fields)*
                                    marker: std::marker::PhantomData,
                                }
                            }
                        }
                    })
                }
                OverType::Float(_) => Impl::External,
            },
            Norm2 => match ty {
                OverType::Type(_) => {
                    let mut bounds = TraitBounds::default();
                    bounds.insert(MagParam::A);
                    bounds.insert(T.copy());
                    let (trait_ty, trait_fn) = self.ty_fn();
                    let ty_t = ty.with_type_param(T, MagParam::A);
                    let scalar = Type::Grade(0);
                    let scalar_ty = Type::Grade(0).with_type_param(T, MagParam::A);
                    let mut sum =
                        TypeFields::new(algebra, ty).fold(quote!(), |mut ts, (b, field)| {
                            let product = algebra.dot(b, b.rev());
                            if product.is_zero() {
                                return ts;
                            } else {
                                bounds.insert(T.mul(T, T));
                            }
                            if product.is_positive() {
                                if ts.is_empty() {
                                    quote!(self.#field * self.#field)
                                } else {
                                    bounds.insert(T.add(T, T));
                                    quote!(+ self.#field * self.#field)
                                }
                                .to_tokens(&mut ts)
                            } else if product.is_negative() {
                                if ts.is_empty() {
                                    bounds.insert(T.neg());
                                    quote!(-(self.#field * self.#field))
                                } else {
                                    bounds.insert(T.sub(T, T));
                                    quote!(- self.#field * self.#field)
                                }
                                .to_tokens(&mut ts)
                            }
                            ts
                        });

                    if sum.is_empty() {
                        bounds.insert(T.zero());
                        let (zero_ty, zero_fn) = UnaryTrait::Zero.ty_fn();
                        quote!(#zero_ty::#zero_fn()).to_tokens(&mut sum);
                    }

                    let (params, where_clause) = bounds.params_and_where_clause();
                    Impl::Actual(quote! {
                        impl #params #trait_ty for #ty_t #where_clause {
                            type Output = #scalar_ty;
                            #[inline]
                            fn #trait_fn(self) -> Self::Output {
                                #scalar {
                                    s: #sum,
                                    marker: std::marker::PhantomData,
                                }
                            }
                        }
                    })
                }
                OverType::Float(_) => Impl::None,
            },
            Norm => match ty {
                OverType::Type(_) => {
                    if UnaryTrait::Norm2.no_impl(ty, algebra) {
                        return Impl::None;
                    }
                    let (trait_ty, trait_fn) = self.ty_fn();
                    let (norm2_ty, norm2_fn) = UnaryTrait::Norm2.ty_fn();
                    let ty_t = ty.with_type_param(T, A);
                    let scalar = Type::Grade(0);
                    let scalar_t = Type::Grade(0).with_type_param(T, A);
                    let s = &algebra.fields[Blade(0)];
                    Impl::Actual(quote! {
                        impl<T, A> #trait_ty for #ty_t
                        where
                            #ty_t: #norm2_ty<Output = #scalar_t> + Copy,
                            T: clifford::Number,
                        {
                            type Output = #scalar_t;
                            #[inline]
                            fn #trait_fn(self) -> Self::Output {
                                let s2 = #norm2_ty::#norm2_fn(self).#s;
                                #scalar {
                                    #s: Sqrt::sqrt(s2),
                                    marker: std::marker::PhantomData,
                                }
                            }
                        }
                    })
                }
                OverType::Float(_) => Impl::None,
            },
            Unitize => match ty {
                OverType::Type(ty) => {
                    if UnaryTrait::Norm.no_impl(ty, algebra)
                        || BinaryTrait::Div.no_impl(ty, Type::Grade(0), algebra)
                    {
                        return Impl::None;
                    }

                    for _ in 0..10 {
                        let mut value = Value::gen(ty, algebra);
                        value.unit(algebra);
                        let norm = value.norm(algebra);
                        if (norm - 1.).abs() > 1e-10 {
                            return Impl::None;
                        }
                    }

                    let (trait_ty, trait_fn) = self.ty_fn();
                    let (norm_ty, norm_fn) = UnaryTrait::Norm.ty_fn();
                    let (div_ty, div_fn) = BinaryTrait::Div.ty_fn();
                    let ty_t = ty.with_type_param(T, Mag::Any);
                    let scalar_t = Type::Grade(0).with_type_param(T, Mag::Any);
                    let unit_ty = ty.with_type_param(T, Mag::Unit);
                    Impl::Actual(quote! {
                        impl<T> #trait_ty for #ty_t
                        where
                            #ty_t: #norm_ty<Output = #scalar_t> + #div_ty<#scalar_t, Output = #ty_t> + Copy,
                        {
                            type Output = #unit_ty;
                            #[inline]
                            fn #trait_fn(self) -> Self::Output {
                                let norm = #norm_ty::#norm_fn(self);
                                #div_ty::#div_fn(self, norm).assert()
                            }
                        }
                        impl<T> #trait_ty for #unit_ty {
                            type Output = #unit_ty;
                            #[inline]
                            fn #trait_fn(self) -> Self::Output {
                                self
                            }
                        }
                    })
                }
                _ => Impl::None,
            },
            Inverse => match ty {
                OverType::Type(inner) => {
                    if UnaryTrait::Norm2.no_impl(ty, algebra)
                        || UnaryTrait::Reverse.no_impl(ty, algebra)
                    {
                        return Impl::None;
                    }

                    let blade_count = TypeBlades::new(algebra, ty).count();

                    let (trait_ty, trait_fn) = self.ty_fn();
                    let (rev_ty, rev_fn) = UnaryTrait::Reverse.ty_fn();

                    let inv_ty = {
                        let (one_ty, one_fn) = UnaryTrait::One.ty_fn();
                        let (norm2_ty, norm2_fn) = UnaryTrait::Norm2.ty_fn();
                        let ty_t = inner.with_type_param(T, Mag::Any);

                        let s = &algebra.fields[Blade(0)];

                        let mut bounds = TraitBounds::default();
                        bounds.insert(ty_t.copy());

                        bounds.insert(T.copy());

                        bounds.insert(T.div(T, T));
                        bounds.insert(ty_t.norm2());
                        bounds.insert(ty_t.rev());

                        let (fields, inv_norm2) = if blade_count >= 4 {
                            bounds.insert(T.one());
                            bounds.insert(T.mul(T, T));
                            (
                                TypeFields::new(algebra, inner)
                                    .map(|(_, field)| {
                                        quote! {
                                            #field: rev.#field * inv_norm2
                                        }
                                    })
                                    .collect::<Vec<_>>(),
                                Some(quote!(let inv_norm2 = <T as #one_ty>::#one_fn() / norm2;)),
                            )
                        } else {
                            (
                                TypeFields::new(algebra, inner)
                                    .map(|(_, field)| {
                                        quote! {
                                            #field: rev.#field / norm2
                                        }
                                    })
                                    .collect::<Vec<_>>(),
                                None,
                            )
                        };

                        let (params, where_clause) = bounds.clone().params_and_where_clause();

                        quote! {
                            impl #params #trait_ty for #ty_t #where_clause {
                                type Output = #ty_t;
                                #[inline]
                                fn #trait_fn(self) -> Self::Output {
                                    let rev = #rev_ty::#rev_fn(self);
                                    let norm2 = #norm2_ty::#norm2_fn(self).#s;
                                    #inv_norm2
                                    #inner {
                                        #(#fields,)*
                                        marker: std::marker::PhantomData,
                                    }
                                }
                            }
                        }
                    };

                    let inv_unit = {
                        let unit_t = inner.with_type_param(T, Mag::Unit);
                        let mut bounds = TraitBounds::default();
                        bounds.insert(unit_t.rev());
                        let (params, where_clause) = bounds.params_and_where_clause();
                        quote! {
                            impl #params #trait_ty for #unit_t #where_clause {
                                type Output = #unit_t;
                                #[inline]
                                fn #trait_fn(self) -> Self::Output {
                                    #rev_ty::#rev_fn(self)
                                }
                            }
                        }
                    };

                    Impl::Actual(quote! {
                        #inv_ty
                        #inv_unit
                    })
                }
                OverType::Float(_) => Impl::External,
            },
            Pod => {
                if ty.is_float() {
                    return Impl::External;
                }
                let trait_ty = self.ty();
                let zeroable_ty = UnaryTrait::Zeroable.ty();
                let ty_t = ty.with_type_param(T, MagParam::A);
                Impl::Actual(quote! {
                    unsafe impl<T, A> #trait_ty for #ty_t
                    where
                        T: #zeroable_ty + #trait_ty,
                        Self: Copy + 'static,
                    {}
                })
            }
            Zeroable => {
                // while unit types are not valid in a zeroed state, they are still memory safe
                if ty.is_float() {
                    return Impl::External;
                }
                let ty_t = ty.with_type_param(T, MagParam::A);
                let trait_ty = self.ty();
                Impl::Actual(quote! {
                    unsafe impl<T, A> #trait_ty for #ty_t
                    where
                        T: #trait_ty,
                    {}
                })
            }
            FloatType => {
                if ty.is_float() {
                    return Impl::None;
                }
                let trait_ty = self.ty();
                let ty_t = ty.with_type_param(T, A);
                Impl::Actual(quote! {
                    impl<T, A> #trait_ty for #ty_t {
                        type Float = T;
                    }
                })
            }
            Rand => match ty {
                OverType::Float(_) => Impl::External,
                OverType::Type(Type::Mv) => Impl::None,
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
                            quote!(rand::Rng::gen::<#vec_t>(rng) * rand::Rng::gen::<#vec_t>(rng))
                        }
                        OverType::Type(Type::Flector) => {
                            quote!(rand::Rng::gen::<#vec_t>(rng) * rand::Rng::gen::<#vec_t>(rng) * rand::Rng::gen::<#vec_t>(rng))
                        }
                        _ => unimplemented!(),
                    };
                    let (params, where_clause) = bounds.params_and_where_clause();

                    Impl::Actual(quote! {
                        impl #params rand::distributions::Distribution<#ty_t> for rand::distributions::Standard
                            #where_clause
                                rand::distributions::Standard: rand::distributions::Distribution<#t> + rand::distributions::Distribution<#vec_t>,
                        {
                            #[inline]
                            fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) ->  #ty_t {
                                #expr
                            }
                        }
                    })
                }
                OverType::Type(Type::Grade(_)) => {
                    let inner = Type::from(ty);
                    let t = T;
                    let ty_t = ty.with_type_param(T, Mag::Any);
                    let unit_t = ty.with_type_param(T, Mag::Unit);
                    let (one_ty, one_fn) = UnaryTrait::One.ty_fn();
                    let (zero_ty, zero_fn) = UnaryTrait::Zero.ty_fn();
                    let (norm2_ty, norm2_fn) = UnaryTrait::Norm2.ty_fn();
                    let mut bounds = TraitBounds::default();
                    bounds.insert(t.one());
                    let mut has_non_zero = false;
                    let fields = TypeFields::new(algebra, ty)
                        .map(|(blade, field)| {
                            if algebra.dot(blade, blade).is_zero() {
                                bounds.insert(t.zero());
                                quote!(#field: #zero_ty::#zero_fn())
                            } else {
                                has_non_zero = true;
                                bounds.insert(t.mul(t, t));
                                bounds.insert(t.sub(t, t));
                                quote!(#field: rand::Rng::gen_range(rng, -one..=one))
                            }
                        })
                        .collect::<Vec<_>>();

                    if !has_non_zero {
                        return Impl::None;
                    }

                    let s = &algebra.fields[Blade(0)];

                    let inner_t = inner.with_type_param(t, Mag::Any);
                    bounds.insert(inner_t.norm2());
                    let (params, where_clause) = bounds.params_and_where_clause();

                    Impl::Actual(quote! {
                        impl #params rand::distributions::Distribution<#ty_t> for rand::distributions::Standard
                        #where_clause
                            #t: clifford::Number + rand::distributions::uniform::SampleUniform,
                            std::ops::RangeInclusive<#t>: rand::distributions::uniform::SampleRange<#t>,
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
                        impl #params rand::distributions::Distribution<#unit_t> for rand::distributions::Standard
                        #where_clause
                            #t: clifford::Number + rand::distributions::uniform::SampleUniform,
                            std::ops::RangeInclusive<#t>: rand::distributions::uniform::SampleRange<#t>,
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
                    })
                }
            },
            Copy => {
                if ty.is_float() {
                    Impl::External
                } else {
                    Impl::Internal
                }
            }
        }
    }

    pub fn impl_type(&self, ty: OverType, algebra: &Algebra) -> Impl<()> {
        self.define(ty, algebra).map(|_| ())
    }

    pub fn no_impl<T>(self, ty: T, algebra: &Algebra) -> bool
    where
        OverType: From<T>,
    {
        matches!(self.impl_type(ty.into(), algebra), Impl::None)
    }

    pub fn define_tests(self, ty: Type, algebra: &Algebra) -> Option<TokenStream> {
        if self.no_impl(ty, algebra) {
            return None;
        }
        match self {
            UnaryTrait::Inverse => {
                let product = algebra.product(ty, ty, Algebra::geo)?;
                if BinaryTrait::Sub.no_impl(product, product, algebra)
                    || UnaryTrait::Norm2.no_impl(product, algebra)
                    || UnaryTrait::Rand.no_impl(Type::Grade(1), algebra)
                    || algebra.has_negative_bases()
                {
                    return None;
                }
                let ty_ident = ty.fn_ident();
                let fn_ident = format_ident!("inverse_{ty_ident}");
                let ty_t = ty.with_type_param(Float::F64, Mag::Any);
                let (inv_ty, inv_fn) = UnaryTrait::Inverse.ty_fn();
                let (norm2_ty, norm2_fn) = UnaryTrait::Norm2.ty_fn();
                let s = &algebra.fields[Blade(0)];
                let vec_t = Type::Grade(1).with_type_param(Float::F64, Mag::Any);
                let value_expr = match ty {
                    Type::Grade(_) => quote!(rng.gen::<#ty_t>()),
                    Type::Motor => quote!(rng.gen::<#vec_t>() * rng.gen::<#vec_t>()),
                    Type::Flector => {
                        quote!(rng.gen::<#vec_t>() * rng.gen::<#vec_t>() * rng.gen::<#vec_t>())
                    }
                    Type::Mv => return None,
                };
                Some(quote! {
                    #[test]
                    fn #fn_ident() {
                        use rand::Rng;
                        let mut rng = rand::thread_rng();
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
            UnaryTrait::Unitize => {
                let product = algebra.product(ty, ty, Algebra::geo)?;
                if BinaryTrait::Sub.no_impl(product, product, algebra)
                    || UnaryTrait::Norm2.no_impl(product, algebra)
                    || UnaryTrait::Rand.no_impl(Type::Grade(1), algebra)
                    || UnaryTrait::Reverse.no_impl(ty, algebra)
                    || algebra.has_negative_bases()
                {
                    return None;
                }
                let ty_ident = ty.fn_ident();
                let fn_ident = format_ident!("unitize_{ty_ident}");
                let ty_t = ty.with_type_param(Float::F64, Mag::Unit);
                let (norm2_ty, norm2_fn) = UnaryTrait::Norm2.ty_fn();
                let (rev_ty, rev_fn) = UnaryTrait::Reverse.ty_fn();
                let s = &algebra.fields[Blade(0)];
                let vec_t_unit = Type::Grade(1).with_type_param(Float::F64, Mag::Unit);
                let unit_value_expr = match ty {
                    Type::Grade(_) => quote!(rng.gen::<#ty_t>()),
                    Type::Motor => quote!(rng.gen::<#vec_t_unit>() * rng.gen::<#vec_t_unit>()),
                    Type::Flector => {
                        quote!(rng.gen::<#vec_t_unit>() * rng.gen::<#vec_t_unit>() * rng.gen::<#vec_t_unit>())
                    }
                    Type::Mv => return None,
                };
                Some(quote! {
                    #[test]
                    fn #fn_ident() {
                        use rand::Rng;
                        let mut rng = rand::thread_rng();
                        for _ in 0..100 {
                            let unit = #unit_value_expr;
                            let product = unit * #rev_ty::#rev_fn(unit);
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
            _ => None,
        }
    }
}

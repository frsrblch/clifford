use super::*;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, EnumIter, Hash)]
pub enum BinaryTrait {
    /// Ops
    Add,
    Sub,
    Mul,
    Div,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    Wedge,
    Dot,
    Geo,
    Antiwedge,
    Antidot,
    Antigeo,
    Commutator,
    LeftContraction,
    RightContraction,
    Sandwich,
    Antisandwich,
    /// Overloads
    BitAnd,
    BitOr,
    BitXor,
    Shr,
    /// Misc
    From,
    PartialEq,
    // SubsetOf
    GradeFilter,
}

impl ToTokens for BinaryTrait {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ty().to_tokens(tokens);
    }
}

impl BinaryTrait {
    pub fn ty(&self) -> TokenStream {
        use BinaryTrait::*;
        match self {
            Add => quote!(std::ops::Add),
            Sub => quote!(std::ops::Sub),
            Mul => quote!(std::ops::Mul),
            Div => quote!(std::ops::Div),
            AddAssign => quote!(std::ops::AddAssign),
            SubAssign => quote!(std::ops::SubAssign),
            MulAssign => quote!(std::ops::MulAssign),
            DivAssign => quote!(std::ops::DivAssign),
            Wedge => quote!(clifford::Wedge),
            Dot => quote!(clifford::Dot),
            Geo => quote!(clifford::Geo),
            Antiwedge => quote!(clifford::Antiwedge),
            Antidot => quote!(clifford::Antidot),
            Antigeo => quote!(clifford::Antigeo),
            Commutator => quote!(clifford::Commutator),
            LeftContraction => quote!(clifford::LeftContraction),
            RightContraction => quote!(clifford::RightContraction),
            Sandwich => quote!(clifford::Sandwich),
            Antisandwich => quote!(clifford::Antisandwich),
            BitAnd => quote!(std::ops::BitAnd),
            BitOr => quote!(std::ops::BitOr),
            BitXor => quote!(std::ops::BitXor),
            Shr => quote!(std::ops::Shr),
            From => quote!(std::convert::From),
            PartialEq => quote!(std::cmp::PartialEq),
            GradeFilter => quote!(clifford::GradeFilter),
        }
    }

    pub fn fn_ident(&self) -> TokenStream {
        use BinaryTrait::*;
        match self {
            Add => quote!(add),
            Sub => quote!(sub),
            Mul => quote!(mul),
            Div => quote!(div),
            AddAssign => quote!(add_assign),
            SubAssign => quote!(sub_assign),
            MulAssign => quote!(mul_assign),
            DivAssign => quote!(div_assign),
            Wedge => quote!(wedge),
            Dot => quote!(dot),
            Geo => quote!(geo),
            Antiwedge => quote!(antiwedge),
            Antidot => quote!(antidot),
            Antigeo => quote!(antigeo),
            Commutator => quote!(com),
            LeftContraction => quote!(left_con),
            RightContraction => quote!(right_con),
            Sandwich => quote!(sandwich),
            Antisandwich => quote!(antisandwich),
            BitAnd => quote!(bitand),
            BitOr => quote!(bitor),
            BitXor => quote!(bitxor),
            Shr => quote!(shr),
            From => quote!(from),
            PartialEq => quote!(eq),
            GradeFilter => quote!(grade_filter),
        }
    }

    pub fn ty_fn(self) -> (TokenStream, TokenStream) {
        (self.ty(), self.fn_ident())
    }

    pub fn define(self, lhs: OverType, rhs: OverType, algebra: &Algebra) -> Impl<TokenStream> {
        use BinaryTrait::*;
        use FloatParam::{T, U, V};
        use Mag::Any;
        use MagParam::{A, B};

        if lhs.is_float() && rhs.is_float() {
            return if lhs == rhs {
                match self {
                    Add | Sub | Mul | Div | AddAssign | SubAssign | MulAssign | DivAssign => {
                        Impl::External
                    }
                    _ => Impl::None,
                }
            } else {
                Impl::None
            };
        }

        if lhs.is_float() {
            return Impl::None;
        }

        match self {
            Add | Sub => {
                if lhs.is_float() && rhs.is_float() {
                    return if lhs == rhs {
                        Impl::External
                    } else {
                        Impl::None
                    };
                }
                let Some(output) = Type::from(lhs) + Type::from(rhs) else {
                    return Impl::None;
                };
                let (trait_ty, trait_fn) = self.ty_fn();
                let (from_ty, from_fn) = BinaryTrait::From.ty_fn();
                let Some((mut bounds, [t, u, v], [a, b, c])) = TraitBounds::sum_types(lhs, rhs)
                else {
                    return Impl::None;
                };
                let lhs_t = lhs.with_type_param(t, a);
                let rhs_u = rhs.with_type_param(u, b);
                let output_v = output.with_type_param(v, c);

                let op = if self == Add { quote!(+) } else { quote!(-) };
                let sign = if self == Add { quote!() } else { quote!(-) };
                let zero = quote!(clifford::Zero::zero());
                let fields = TypeFields::new(algebra, output)
                    .map(|(blade, field)| {
                        match (lhs.contains_blade(blade), rhs.contains_blade(blade)) {
                            (true, true) => {
                                if self == Add {
                                    bounds.insert(t.add(u, v));
                                } else {
                                    bounds.insert(t.sub(u, v));
                                }
                                let lf = lhs.access_field(&quote!(self), blade, algebra);
                                let rf = rhs.access_field(&quote!(rhs), blade, algebra);
                                quote!(#field: #lf #op #rf)
                            }
                            (true, false) => {
                                let lf = lhs.access_field(&quote!(self), blade, algebra);
                                if v != t {
                                    bounds.insert(v.from(t));
                                    quote!(#field: #from_ty::#from_fn(#lf))
                                } else {
                                    quote!(#field: #lf)
                                }
                            }
                            (false, true) => {
                                if self == Sub {
                                    bounds.insert(u.neg());
                                }
                                let rf = rhs.access_field(&quote!(rhs), blade, algebra);
                                if v != u {
                                    bounds.insert(v.from(u));
                                    quote!(#field: #from_ty::#from_fn(#sign  #rf))
                                } else {
                                    quote!(#field:#sign  #rf)
                                }
                            }
                            (false, false) => {
                                bounds.insert(v.zero());
                                quote!(#field: #zero)
                            }
                        }
                    })
                    .collect::<Vec<_>>();

                let (params, where_clause) = bounds.params_and_where_clause();
                Impl::Actual(quote! {
                    impl #params #trait_ty<#rhs_u> for #lhs_t #where_clause {
                        type Output = #output_v;
                        #[inline]
                        fn #trait_fn(self, rhs: #rhs_u) -> Self::Output {
                            #output {
                                #(#fields,)*
                                marker: std::marker::PhantomData,
                            }
                        }
                    }
                })
            }
            AddAssign | SubAssign => {
                if Some(Type::from(lhs)) != Type::from(lhs) + Type::from(rhs) {
                    return Impl::None;
                }

                let (trait_ty, trait_fn) = self.ty_fn();

                let mut bounds = TraitBounds::default();
                let (t, a) = match lhs {
                    OverType::Type(_) => (T, A),
                    OverType::Float(f) => (f.into(), Any.into()),
                };
                let (u, b) = match rhs {
                    OverType::Type(_) => (T, B),
                    OverType::Float(f) => (f.into(), Any.into()),
                };
                bounds.insert([t, u]);
                bounds.insert([a, b]);
                match self {
                    AddAssign => bounds.insert(t.add_assign(u)),
                    SubAssign => bounds.insert(t.sub_assign(u)),
                    _ => {}
                }

                let lhs_t = lhs.with_type_param(t, a);
                let rhs_t = rhs.with_type_param(u, b);

                let fields = TypeBlades::new(algebra, lhs).filter_map(|blade| {
                    if rhs.contains_blade(blade) {
                        let lhs_access = lhs.access_field(&quote!(self), blade, algebra);
                        let rhs_access = rhs.access_field(&quote!(rhs), blade, algebra);
                        if lhs.is_float() {
                            Some(quote! {
                                #trait_ty::#trait_fn(#lhs_access, #rhs_access);
                            })
                        } else {
                            Some(quote! {
                                #trait_ty::#trait_fn(&mut #lhs_access, #rhs_access);
                            })
                        }
                    } else {
                        None
                    }
                });

                let (params, where_clause) = bounds.params_and_where_clause();
                Impl::Actual(quote! {
                    impl #params #trait_ty<#rhs_t> for #lhs_t #where_clause {
                        #[inline]
                        fn #trait_fn(&mut self, rhs: #rhs_t) {
                            #(#fields)*
                        }
                    }
                })
            }
            Mul | Geo | Dot | Wedge | Antigeo | Antidot | Antiwedge | Commutator
            | LeftContraction | RightContraction | BitAnd | BitOr | BitXor => {
                if (lhs.is_float() || rhs.is_float())
                    && matches!(
                        self,
                        Geo | Dot
                            | Wedge
                            | Antigeo
                            | Antidot
                            | Antiwedge
                            | Commutator
                            | LeftContraction
                            | RightContraction
                            | BitAnd
                            | BitOr
                            | BitXor
                    )
                {
                    return Impl::None;
                }

                // prevents downstream crates from infinte type recursion
                if lhs.is_float() && self == Mul {
                    return Impl::None;
                }

                let (trait_ty, trait_fn) = self.ty_fn();
                let (zero_ty, zero_fn) = UnaryTrait::Zero.ty_fn();

                let Some(output) = self.product(lhs, rhs, algebra) else {
                    return Impl::None;
                };

                let bounds_and_types = if matches!(self, Mul | Geo) {
                    TraitBounds::geo_types(lhs, rhs, algebra, self)
                } else {
                    TraitBounds::product_types(lhs, rhs)
                };
                let Some((mut bounds, [t, u, v], [a, b, c])) = bounds_and_types else {
                    return Impl::None;
                };
                let lhs_t = lhs.with_type_param(t, a);
                let rhs_u = rhs.with_type_param(u, b);
                let output_v = output.with_type_param(v, c);

                let self_var = &quote!(self);
                let rhs_var = &quote!(rhs);
                let fields = TypeFields::new(algebra, output)
                    .map(|(blade, field)| {
                        let mut sum = quote!();
                        for (l, r) in algebra.blade_tuples(lhs, rhs) {
                            let out = match self {
                                Mul | Geo => algebra.geo(l, r),
                                Dot | BitOr => algebra.dot(l, r),
                                Wedge | BitXor => algebra.wedge(l, r),
                                Antigeo => algebra.antigeo(l, r),
                                Antidot => algebra.antidot(l, r),
                                Antiwedge | BitAnd => algebra.antiwedge(l, r),
                                Commutator => algebra.commutator(l, r),
                                LeftContraction => algebra.left_con(l, r),
                                RightContraction => algebra.right_con(l, r),
                                _ => unimplemented!(),
                            };
                            let lv = lhs.access_field(self_var, l, algebra);
                            let rv = rhs.access_field(rhs_var, r, algebra);
                            if out == blade {
                                if sum.is_empty() {
                                    bounds.insert(t.mul(u, v));
                                    quote!(#lv * #rv)
                                } else {
                                    bounds.insert(t.mul(u, v));
                                    bounds.insert(v.add(v, v));
                                    quote!(+ #lv * #rv)
                                }
                                .to_tokens(&mut sum);
                            } else if out == -blade {
                                if sum.is_empty() {
                                    bounds.insert(t.mul(u, v));
                                    bounds.insert(v.neg());
                                    quote!(-(#lv * #rv))
                                } else {
                                    bounds.insert(t.mul(u, v));
                                    bounds.insert(v.sub(v, v));
                                    quote!(- #lv * #rv)
                                }
                                .to_tokens(&mut sum);
                            }
                        }
                        if sum.is_empty() {
                            bounds.insert(v.zero());
                            quote!(#field: #zero_ty::#zero_fn())
                        } else {
                            quote!(#field: #sum)
                        }
                    })
                    .collect::<Vec<_>>();

                let (params, where_clause) = bounds.params_and_where_clause();

                let expr = match output {
                    OverType::Type(ty) => quote! {
                        #ty {
                            #(#fields,)*
                            marker: std::marker::PhantomData,
                        }
                    },
                    OverType::Float(_) => unimplemented!(),
                };

                let suspicious_arithmetic_impl = if matches!(self, BitAnd | BitOr | BitXor) {
                    quote!(#[allow(clippy::suspicious_arithmetic_impl)])
                } else {
                    quote!()
                };

                Impl::Actual(quote! {
                    impl #params #trait_ty<#rhs_u> for #lhs_t #where_clause {
                        type Output = #output_v;
                        #[inline]
                        #suspicious_arithmetic_impl
                        fn #trait_fn(self, rhs: #rhs_u) -> Self::Output {
                            #expr
                        }
                    }
                })
            }
            Div => {
                if UnaryTrait::Inverse.no_impl(rhs, algebra) {
                    return Impl::None;
                }
                let (trait_ty, trait_fn) = self.ty_fn();
                let (inv_ty, inv_fn) = UnaryTrait::Inverse.ty_fn();
                let (zero_ty, zero_fn) = UnaryTrait::Zero.ty_fn();
                let Some((mut bounds, [t, u, v], [a, b, c])) =
                    TraitBounds::geo_types(lhs, rhs, algebra, self)
                else {
                    return Impl::None;
                };

                let lhs_t = lhs.with_type_param(t, a);
                let rhs_u = rhs.with_type_param(u, b);
                let Some(output) = algebra.product(lhs, rhs, Algebra::geo) else {
                    return Impl::None;
                };
                let output_v = output.with_type_param(v, c);
                let self_var = &quote!(self);

                let generate_with_div = |mut bounds: TraitBounds| {
                    let rhs_var = &quote!(rhs);
                    let fields = TypeFields::new(algebra, output)
                        .map(|(blade, field)| {
                            let mut sum = quote!();
                            for (l, r) in algebra.blade_tuples(lhs, rhs) {
                                let out = algebra.geo(l, r);
                                let lv = lhs.access_field(self_var, l, algebra);
                                let rv = rhs.access_field(rhs_var, r, algebra);
                                if out == blade {
                                    if sum.is_empty() {
                                        bounds.insert(t.div(u, v));
                                        quote!(#lv / #rv)
                                    } else {
                                        bounds.insert(t.div(u, v));
                                        bounds.insert(v.add(v, v));
                                        quote!(+ #lv / #rv)
                                    }
                                    .to_tokens(&mut sum);
                                } else if out == -blade {
                                    if sum.is_empty() {
                                        bounds.insert(t.div(u, v));
                                        bounds.insert(v.neg());
                                        quote!(-(#lv / #rv))
                                    } else {
                                        bounds.insert(t.div(u, v));
                                        bounds.insert(v.sub(v, v));
                                        quote!(- #lv / #rv)
                                    }
                                    .to_tokens(&mut sum);
                                }
                            }
                            if sum.is_empty() {
                                bounds.insert(V.zero());
                                quote!(#field: #zero_ty::#zero_fn())
                            } else {
                                quote!(#field: #sum)
                            }
                        })
                        .collect::<Vec<_>>();
                    let (params, where_clause) = bounds.params_and_where_clause();
                    Impl::Actual(quote! {
                        impl #params #trait_ty<#rhs_u> for #lhs_t #where_clause {
                            type Output = #output_v;
                            #[inline]
                            #[allow(clippy::suspicious_arithmetic_impl)]
                            fn #trait_fn(self, rhs: #rhs_u) -> Self::Output {
                                #output {
                                    #(#fields,)*
                                    marker: std::marker::PhantomData,
                                }
                            }
                        }
                    })
                };

                // TODO use division for single bladed types
                match rhs {
                    OverType::Type(rhs_ty) if algebra.type_blades(rhs_ty).count() == 1 => {
                        generate_with_div(bounds)
                    }
                    OverType::Float(_) => generate_with_div(bounds),
                    _ => {
                        let inv_var = &quote!(inv);
                        bounds.insert(rhs_u.inv());
                        let fields = TypeFields::new(algebra, output)
                            .map(|(blade, field)| {
                                let mut sum = quote!();
                                for (l, r) in algebra.blade_tuples(lhs, rhs) {
                                    let out = algebra.geo(l, r);
                                    let lv = lhs.access_field(self_var, l, algebra);
                                    let rv = rhs.access_field(inv_var, r, algebra);
                                    if out == blade {
                                        if sum.is_empty() {
                                            bounds.insert(t.mul(u, v));
                                            quote!(#lv * #rv)
                                        } else {
                                            bounds.insert(t.mul(u, v));
                                            bounds.insert(v.add(v, v));
                                            quote!(+ #lv * #rv)
                                        }
                                        .to_tokens(&mut sum);
                                    } else if out == -blade {
                                        if sum.is_empty() {
                                            bounds.insert(t.mul(u, v));
                                            bounds.insert(v.neg());
                                            quote!(-(#lv * #rv))
                                        } else {
                                            bounds.insert(t.mul(u, v));
                                            bounds.insert(v.sub(v, v));
                                            quote!(- #lv * #rv)
                                        }
                                        .to_tokens(&mut sum);
                                    }
                                }
                                if sum.is_empty() {
                                    bounds.insert(V.zero());
                                    quote!(#field: #zero_ty::#zero_fn())
                                } else {
                                    quote!(#field: #sum)
                                }
                            })
                            .collect::<Vec<_>>();
                        let (params, where_clause) = bounds.params_and_where_clause();
                        Impl::Actual(quote! {
                            impl #params #trait_ty<#rhs_u> for #lhs_t #where_clause {
                                type Output = #output_v;
                                #[inline]
                                #[allow(clippy::suspicious_arithmetic_impl)]
                                fn #trait_fn(self, rhs: #rhs_u) -> Self::Output {
                                    let inv = #inv_ty::#inv_fn(rhs);
                                    #output {
                                        #(#fields,)*
                                        marker: std::marker::PhantomData,
                                    }
                                }
                            }
                        })
                    }
                }
            }
            MulAssign | DivAssign => {
                let Some(output) = BinaryTrait::Mul.product(lhs, rhs, algebra) else {
                    return Impl::None;
                };
                if lhs != output {
                    return Impl::None;
                }
                let (trait_ty, trait_fn) = self.ty_fn();
                let inner = match self {
                    MulAssign => Mul,
                    DivAssign => Div,
                    _ => unreachable!(),
                };
                if inner.no_impl(lhs, rhs, algebra) {
                    return Impl::None;
                }
                let (inner_ty, inner_fn) = inner.ty_fn();
                let mut bounds = TraitBounds::default();
                let [t, u, v] = match (lhs, rhs) {
                    (OverType::Float(_), OverType::Float(_)) => unimplemented!(),
                    (OverType::Float(f), _) => [FloatParam::Float(f), T, T],
                    (_, OverType::Float(f)) => [T, FloatParam::Float(f), T],
                    _ => [T, T, T],
                };
                bounds.insert([t, u, v]);
                let [a, b, c] = match (lhs, rhs) {
                    (OverType::Float(_), OverType::Float(_)) => [Any.into(); 3],
                    (OverType::Float(_), _) => [Any.into(), B, Any.into()],
                    (_, OverType::Float(_)) => [Any.into(); 3],
                    _ => {
                        if self == MulAssign {
                            bounds.insert(A.mul(B, A));
                        }
                        if self == DivAssign {
                            bounds.insert(A.div(B, A));
                        }
                        [A, B, A]
                    }
                };
                bounds.insert([a, b, c]);

                let lhs_t = lhs.with_type_param(t, a);
                let rhs_u = rhs.with_type_param(u, b);

                let (params, _) = bounds.params_and_where_clause();

                Impl::Actual(quote! {
                    impl #params #trait_ty<#rhs_u> for #lhs_t
                    where
                        #lhs_t: #inner_ty<#rhs_u, Output = #lhs_t> + Copy
                    {
                        fn #trait_fn(&mut self, rhs: #rhs_u) {
                            *self = #inner_ty::#inner_fn(*self, rhs);
                        }
                    }
                })
            }
            From => {
                if lhs.is_float() {
                    if lhs == rhs {
                        return Impl::External;
                    } else {
                        return Impl::None;
                    }
                }
                if !lhs.contains(rhs) {
                    return Impl::None;
                }
                let (t, a) = if let OverType::Float(f) = lhs {
                    (FloatParam::Float(f), Any.into())
                } else if let OverType::Float(f) = rhs {
                    (FloatParam::Float(f), Any.into())
                } else {
                    (T, A)
                };
                let lhs_ty = Type::from(lhs);
                let rhs_ty = Type::from(rhs);
                let (from_ty, from_fn) = From.ty_fn();
                let (zero_ty, zero_fn) = UnaryTrait::Zero.ty_fn();
                let mut bounds = TraitBounds::from_iter([t]);

                let (lhs_t, rhs_t) = if lhs == rhs {
                    (
                        lhs.with_type_param(t, Mag::Any),
                        rhs.with_type_param(t, Mag::Unit),
                    )
                } else {
                    bounds.insert(a);
                    (lhs.with_type_param(t, a), rhs.with_type_param(t, a))
                };

                let var = quote!(value);
                let s = &algebra.fields[Blade(0)];
                let expr = {
                    let fields = TypeFields::new(algebra, lhs).map(|(b, f)| {
                        if rhs_ty.contains(b) {
                            let rhs_field = rhs.access_field(&var, b, algebra);
                            quote! {
                                #f: #rhs_field
                            }
                        } else {
                            if t.is_generic() {
                                bounds.insert(t.zero());
                            }
                            quote! {
                                #f: #zero_ty::#zero_fn()
                            }
                        }
                    });
                    match lhs {
                        OverType::Type(_) => {
                            quote! {
                                #lhs_ty {
                                    #(#fields,)*
                                    marker: std::marker::PhantomData,
                                }
                            }
                        }
                        OverType::Float(_) => {
                            quote! {
                                #var.#s
                            }
                        }
                    }
                };
                let (params, where_clause) = bounds.params_and_where_clause();
                Impl::Actual(quote! {
                    impl #params #from_ty<#rhs_t> for #lhs_t #where_clause {
                        fn #from_fn(#var: #rhs_t) -> Self {
                            #expr
                        }
                    }
                })
            }
            Sandwich | Antisandwich | Shr => {
                // TODO rework with SubsetOf traits rather than GradeProduct
                match Type::from(lhs) {
                    Type::Mv | Type::Grade(0) => return Impl::None,
                    Type::Grade(g) if g == algebra.dim() => return Impl::None,
                    _ => {}
                }
                match Type::from(rhs) {
                    Type::Grade(0) => return Impl::None,
                    Type::Grade(g) if g == algebra.dim() => return Impl::None,
                    _ => {}
                }

                let Some(int) = algebra.product(lhs, rhs, Algebra::geo) else {
                    return Impl::None;
                };
                let Some(raw_out) = algebra.product(int, lhs, Algebra::geo) else {
                    return Impl::None;
                };

                let out = match (Type::from(rhs), raw_out) {
                    (Type::Grade(r), _) | (_, Type::Grade(r)) => Type::Grade(r),
                    (rhs, _) => rhs,
                };

                let (trait_ty, trait_fn) = self.ty_fn();
                let (rev_ty, rev_fn) = UnaryTrait::Reverse.ty_fn();
                let (geo_ty, geo_fn) = BinaryTrait::Geo.ty_fn();
                let t = FloatParam::T;
                let [a, b, c, d] = [MagParam::A, MagParam::B, MagParam::C, MagParam::D];

                let lhs_t = lhs.with_type_param(t, a);
                let rhs_t = rhs.with_type_param(t, b);
                let int_t = int.with_type_param(t, c);
                let raw_out_t = raw_out.with_type_param(t, d);
                let out_t = out.with_type_param(t, b);

                let rev = quote!(#rev_ty::#rev_fn(self));
                let intermediate = quote!(#geo_ty::#geo_fn(self, rhs));
                let (bound, expr) = match (raw_out, out) {
                    (raw_out, out) if raw_out == out => (
                        quote!(#int_t: #geo_ty<#lhs_t, Output = #raw_out_t>),
                        quote!(#geo_ty::#geo_fn(#intermediate, #rev).assert()),
                    ),
                    (_, out) => {
                        let fn_ident = out.fn_ident();
                        (
                            quote!(#int_t: #geo_ty<#lhs_t, Output = #raw_out_t>),
                            quote!(#geo_ty::#geo_fn(#intermediate, #rev).#fn_ident().assert()),
                        )
                    }
                };

                Impl::Actual(quote! {
                    impl<T, A, B, C, D> #trait_ty<#rhs_t> for #lhs_t
                    where
                        #lhs_t: #rev_ty<Output = #lhs_t>
                            + #geo_ty<#rhs_t, Output = #int_t>
                            + Copy,
                        #bound
                    {
                        type Output = #out_t;
                        #[inline]
                        fn #trait_fn(self, rhs: #rhs_t) -> Self::Output {
                            #expr
                        }
                    }
                })
            }
            PartialEq => {
                if lhs == rhs {
                    if lhs.is_float() {
                        Impl::External
                    } else {
                        let (trait_ty, trait_fn) = self.ty_fn();
                        let mut bounds = TraitBounds::default();
                        bounds.insert(T.partial_eq(T));
                        bounds.insert([A, B]);
                        let lhs_t = lhs.with_type_param(T, A);
                        let rhs_t = rhs.with_type_param(T, B);
                        let cmp_fields =
                            TypeFields::new(algebra, lhs).fold(quote!(), |mut ts, (_, f)| {
                                let expr = quote!(self.#f.eq(&rhs.#f));
                                if ts.is_empty() { expr } else { quote!(& #expr) }
                                    .to_tokens(&mut ts);
                                ts
                            });
                        let (params, where_clause) = bounds.params_and_where_clause();
                        Impl::Actual(quote! {
                            impl #params #trait_ty<#rhs_t> for #lhs_t #where_clause {
                                fn #trait_fn(&self, rhs: &#rhs_t) -> bool {
                                    #cmp_fields
                                }
                            }
                        })
                    }
                } else {
                    Impl::None
                }
            }
            GradeFilter => {
                if !rhs.contains(lhs) || lhs.is_float() || rhs.is_float() {
                    return Impl::None;
                }
                let OverType::Type(lhs) = lhs else {
                    return Impl::None;
                };
                let OverType::Type(rhs) = rhs else {
                    return Impl::None;
                };
                let (trait_ty, trait_fn) = self.ty_fn();
                let lhs_t = lhs.with_type_param(T, A);
                let out_t = if TypeGrades::new(algebra, lhs).all(|g| rhs.contains_grade(g)) {
                    rhs.with_type_param(T, A)
                } else {
                    rhs.with_type_param(T, Mag::Any)
                };
                let rhs_u = rhs.with_type_param(U, B);
                let mut bounds = TraitBounds::default();
                bounds.insert([T, U]);
                bounds.insert([A, B]);
                let fields = TypeFields::new(algebra, rhs).map(|(b, f)| {
                    if lhs.contains_grade(b.grade()) {
                        quote!(#f: self.#f)
                    } else {
                        bounds.insert(T.zero());
                        quote!(#f: T::zero())
                    }
                });
                let expr = if lhs == rhs {
                    quote!(self)
                } else {
                    quote! {
                        #rhs {
                            #(#fields,)*
                            marker: std::marker::PhantomData,
                        }
                    }
                };
                let (params, where_clause) = bounds.params_and_where_clause();
                Impl::Actual(quote! {
                    impl #params #trait_ty<#rhs_u> for #lhs_t #where_clause {
                        type Output = #out_t;
                        #[inline]
                        fn #trait_fn(self, _target: #rhs_u) -> Self::Output {
                            #expr
                        }
                    }
                })
            }
        }
    }

    pub fn impl_type(&self, lhs: OverType, rhs: OverType, algebra: &Algebra) -> Impl<()> {
        self.define(lhs, rhs, algebra).map(|_| ())
    }

    pub fn no_impl<T, U>(self, lhs: T, rhs: U, algebra: &Algebra) -> bool
    where
        OverType: From<T> + From<U>,
    {
        matches!(self.impl_type(lhs.into(), rhs.into(), algebra), Impl::None)
    }

    pub fn product(self, lhs: OverType, rhs: OverType, algebra: &Algebra) -> Option<OverType> {
        use BinaryTrait::*;
        match self {
            Mul | Geo => algebra.product(lhs, rhs, Algebra::geo).map(OverType::Type),
            Dot | BitOr => algebra.product(lhs, rhs, Algebra::dot).map(OverType::Type),
            Wedge | BitXor => algebra
                .product(lhs, rhs, Algebra::wedge)
                .map(OverType::Type),
            Antigeo => algebra
                .product(lhs, rhs, Algebra::antigeo)
                .map(OverType::Type),
            Antidot => algebra
                .product(lhs, rhs, Algebra::antidot)
                .map(OverType::Type),
            Antiwedge | BitAnd => algebra
                .product(lhs, rhs, Algebra::antiwedge)
                .map(OverType::Type),
            Commutator => algebra
                .product(lhs, rhs, Algebra::commutator)
                .map(OverType::Type),
            LeftContraction => algebra
                .product(lhs, rhs, Algebra::left_con)
                .map(OverType::Type),
            RightContraction => algebra
                .product(lhs, rhs, Algebra::right_con)
                .map(OverType::Type),
            _ => unimplemented!(),
        }
    }

    pub fn define_tests(
        self,
        lhs: OverType,
        rhs: OverType,
        algebra: &Algebra,
    ) -> Option<TokenStream> {
        match self {
            BinaryTrait::Geo => match (lhs, rhs) {
                (OverType::Type(lhs_inner), OverType::Type(rhs_inner)) => {
                    if UnaryTrait::Unitize.no_impl(lhs, algebra)
                        || UnaryTrait::Unitize.no_impl(rhs, algebra)
                        || UnaryTrait::Norm2.no_impl(lhs, algebra)
                        || UnaryTrait::Norm2.no_impl(rhs, algebra)
                        || BinaryTrait::Mul.no_impl(lhs, rhs, algebra)
                        || algebra.has_negative_bases()
                    {
                        return None;
                    }

                    {
                        let lhs = Value::try_gen_unit(lhs_inner, algebra)?;
                        let rhs = Value::try_gen_unit(rhs_inner, algebra)?;
                        let product = lhs.mul(&rhs, algebra)?;
                        if !product.is_finite() || !product.is_unit(algebra) {
                            return None;
                        }
                    }

                    let lhs_inner = Type::from(lhs);
                    let lhs_lower = lhs_inner.fn_ident();
                    let rhs_inner = Type::from(rhs);
                    let rhs_lower = rhs_inner.fn_ident();
                    let (norm2_ty, norm2_fn) = UnaryTrait::Norm2.ty_fn();

                    let (unit_ty, unit_fn) = UnaryTrait::Unitize.ty_fn();

                    let fn_ident = format_ident!("unit_{lhs_lower}_mul_{rhs_lower}");

                    let lhs_fields = TypeFields::new(algebra, lhs_inner).map(|(_, field)| {
                        quote! {
                            #field: rand::Rng::gen_range(&mut rng, -1.0..1.0),
                        }
                    });

                    let rhs_fields = TypeFields::new(algebra, rhs_inner).map(|(_, field)| {
                        quote! {
                            #field: rand::Rng::gen_range(&mut rng, -1.0..1.0),
                        }
                    });

                    let s = &algebra.fields[Blade(0)];

                    Some(quote! {
                        #[test]
                        fn #fn_ident() {
                            let mut rng = rand::thread_rng();
                            for _ in 0..100 {
                                let lhs = #unit_ty::#unit_fn(#lhs_inner::<f64, Any> {
                                    #(#lhs_fields)*
                                    marker: std::marker::PhantomData,
                                });
                                let rhs = #unit_ty::#unit_fn(#rhs_inner::<f64, Any> {
                                    #(#rhs_fields)*
                                    marker: std::marker::PhantomData,
                                });
                                let product = lhs * rhs;
                                let norm2 = #norm2_ty::#norm2_fn(product).#s.abs();
                                assert!((norm2 - 1f64).abs() < 1e-10, "{}", norm2);
                            }
                        }
                    })
                }
                _ => None,
            },
            _ => None,
        }
    }
}

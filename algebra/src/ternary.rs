use super::*;

// TODO can GradeProduct be replaced by grade functions on types? SubsetOf<SuperSet>::from_superset()?

#[derive(Debug, Copy, Clone, Eq, PartialEq, EnumIter)]
pub enum TernaryTrait {
    GradeProduct,
    GradeAntiproduct,
}

impl TernaryTrait {
    pub fn ty_and_fn(self) -> (TokenStream, TokenStream) {
        (self.ty(), self.fn_ident())
    }

    pub fn ty(self) -> TokenStream {
        match self {
            Self::GradeProduct => quote!(clifford::GradeProduct),
            Self::GradeAntiproduct => quote!(clifford::GradeAntiproduct),
        }
    }

    pub fn fn_ident(self) -> TokenStream {
        match self {
            Self::GradeProduct => quote!(product),
            Self::GradeAntiproduct => quote!(antiproduct),
        }
    }

    pub fn define(
        self,
        lhs: OverType,
        rhs: OverType,
        output: Type,
        algebra: &Algebra,
    ) -> Impl<TokenStream> {
        use FloatParam::*;

        if lhs.is_float() || rhs.is_float() {
            return Impl::None;
        }

        let Some((mut bounds, [t, u, v], [a, b, c])) = TraitBounds::product_types(lhs, rhs) else { return Impl::None };
        bounds.insert(t.mul(u, v));

        let Type::Grade(output_grade) = output else { return Impl::None };
        let lhs_t = lhs.with_type_param(T, a);
        let rhs_u = rhs.with_type_param(U, b);
        let output_v = output.with_type_param(V, c);

        let blades = algebra
            .blade_tuples(lhs, rhs)
            .map(|(l, r)| (l, r, self.product(l, r, algebra)))
            .filter(|(_, _, o)| !o.is_zero() && o.grade() == output_grade)
            .collect::<Vec<_>>();

        if blades.is_empty() {
            return Impl::None;
        }

        let (trait_ty, trait_fn) = self.ty_and_fn();
        let (zero_ty, zero_fn) = UnaryTrait::Zero.ty_fn();

        let fields = SortedTypeBlades::new(algebra, output)
            .map(|blade| (blade, &algebra.fields[blade]))
            .map(|(blade, field)| {
                let mut sum = quote!();
                for &(l, r, o) in blades.iter() {
                    let lf = lhs.access_field(&quote!(lhs), l, algebra);
                    let rf = rhs.access_field(&quote!(rhs), r, algebra);

                    if o == blade {
                        if sum.is_empty() {
                            quote!(#lf * #rf)
                        } else {
                            bounds.insert(V.add(V, V));
                            quote!(+ #lf * #rf)
                        }
                        .to_tokens(&mut sum);
                    } else if -o == blade {
                        if sum.is_empty() {
                            bounds.insert(V.neg());
                            quote!(-(#lf * #rf))
                        } else {
                            bounds.insert(V.sub(V, V));
                            quote!(- #lf * #rf)
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
            impl #params #trait_ty<#lhs_t, #rhs_u> for #output_v #where_clause {
                type Output = #output_v;
                #[inline]
                fn #trait_fn(lhs: #lhs_t, rhs: #rhs_u) -> Self::Output {
                    #output {
                        #(#fields,)*
                        marker: std::marker::PhantomData,
                    }
                }
            }
        })
    }

    pub fn impl_type(
        self,
        lhs: OverType,
        rhs: OverType,
        output: Type,
        algebra: &Algebra,
    ) -> Impl<()> {
        self.define(lhs, rhs, output, algebra).map(|_| ())
    }

    pub fn no_impl(self, lhs: OverType, rhs: OverType, output: Type, algebra: &Algebra) -> bool {
        matches!(self.impl_type(lhs, rhs, output, algebra), Impl::None)
    }

    pub fn product(self, lhs: Blade, rhs: Blade, algebra: &Algebra) -> Blade {
        match self {
            Self::GradeProduct => algebra.geo(lhs, rhs),
            Self::GradeAntiproduct => algebra.antigeo(lhs, rhs),
        }
    }
}

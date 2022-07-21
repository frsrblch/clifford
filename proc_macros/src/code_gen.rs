use crate::algebra::*;
use itertools::iproduct;
use proc_macro2::{Ident, TokenStream};
use std::iter::once;
use syn::token::{Brace, For, Impl};
use syn::{
    Expr, GenericParam, Generics, ImplItem, ItemImpl, ItemStruct, ItemTrait, WherePredicate,
};

// reimplemented because the syn version doesn't track caller
macro_rules! parse_quote {
    ($($tt:tt)*) => {
        syn::parse2(quote::quote!($($tt)*)).unwrap()
    };
}

// TODO Multivector::grade(self) -> Grade
// TODO sandwich product that returns identical grades (this is challenging for multivectors)
//        perhaps have a GradeFilter<Input>
//          Zero: GradeFilter<_, Output = Zero>,
//          Vector: GradeFilter<Vector, Output = Vector>,
//          allows the compiler to eliminate calculations that are filtered out

trait Convert {
    fn convert<U: syn::parse::Parse>(&self) -> U;
}

impl<T: ToTokens> Convert for T {
    #[track_caller]
    fn convert<U: syn::parse::Parse>(&self) -> U {
        syn::parse2(self.to_token_stream()).unwrap()
    }
}

#[allow(dead_code)]
pub fn impl_const(item_impl: &ItemImpl) -> TokenStream {
    let ItemImpl {
        attrs,
        defaultness,
        unsafety,
        impl_token,
        generics,
        trait_,
        self_ty,
        brace_token: _,
        items,
    } = item_impl;
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let (bang, path, for_) = trait_.as_ref().expect("item must be trait impl");
    let trait_ = quote!(#bang #path #for_);
    quote! {
        #(#attrs)*
        #defaultness #unsafety #impl_token #impl_generics const #trait_ #self_ty
        #where_clause
        {
            #(#items)*
        }
    }
}

impl Algebra {
    pub fn define(self) -> TokenStream {
        let product_ops_definitions = ProductOp::iter_all(self).filter_map(|t| t.define());
        let product_ops_blanket_impls =
            ProductOp::iter_all(self).filter_map(|t| t.blanket_impl(self));

        let unary_ops_definitions = UnaryOp::iter(self).filter_map(|t| t.define());
        let unary_ops_blanket_impls = UnaryOp::iter(self).filter_map(|t| t.blanket_impl(self));

        let sum_ops_definitions = SumOp::iter().filter_map(|t| t.define());

        let types = AlgebraType::iter(self).filter_map(AlgebraType::define);

        let type_impls = AlgebraType::iter(self).filter_map(AlgebraType::impl_item);

        let sum_ops = SumOp::iter()
            .flat_map(|op| AlgebraType::iter(self).map(move |lhs| (op, lhs)))
            .flat_map(|(op, lhs)| AlgebraType::iter(self).map(move |rhs| (op, lhs, rhs)))
            .flat_map(|(op, lhs, rhs)| op.item_impl(lhs, rhs));

        let product_ops = ProductOp::iter_all(self)
            .flat_map(|op| AlgebraType::iter(self).map(move |lhs| (op, lhs)))
            .flat_map(|(op, lhs)| AlgebraType::iter(self).map(move |rhs| (op, lhs, rhs)))
            .flat_map(|(op, lhs, rhs)| op.item_impl(lhs, rhs));

        let unary_ops = UnaryOp::iter(self)
            .flat_map(|op| AlgebraType::iter(self).map(move |lhs| (op, lhs)))
            .flat_map(|(op, ty)| op.impl_item(ty));

        let norm_ops = NormOps::iter(self).map(NormOps::define_and_blanket);

        let float_conversion_definitions = FloatConvert::iter().map(FloatConvert::define);

        let float_conversion = AlgebraType::iter(self)
            .flat_map(|ty| FloatConvert::iter().map(move |fc| fc.impl_item(ty)));

        quote! {
            #(#product_ops_definitions)*
            #(#product_ops_blanket_impls)*
            #(#unary_ops_definitions)*
            #(#unary_ops_blanket_impls)*
            #(#sum_ops_definitions)*
            #(#float_conversion_definitions)*
            #(#norm_ops)*
            #(#types)*
            #(#type_impls)*
            #(#sum_ops)*
            #(#product_ops)*
            #(#unary_ops)*
            #(#float_conversion)*
        }
    }
}

impl UnaryOp {
    pub fn define(self) -> Option<ItemTrait> {
        if let UnaryOp::Neg = self {
            return None;
        }

        let ident = self.trait_ty();
        let trait_fn = self.trait_fn();

        Some(parse_quote! {
            pub trait #ident {
                type Output;
                fn #trait_fn(self) -> Self::Output;
            }
        })
    }

    pub fn blanket_impl(self, algebra: Algebra) -> Option<ItemImpl> {
        if let UnaryOp::Antireverse = self {
            let trait_ = self.trait_ty();
            let fn_ = self.trait_fn();

            let mut generics = Generics::default();
            generics.params.push(parse_quote!(T));
            generics.params.push(parse_quote!(U));
            let left_comp = UnaryOp::left_comp(algebra);
            let right_comp = UnaryOp::right_comp(algebra);
            let rev_ty = UnaryOp::Reverse.trait_ty();
            let rev_fn = UnaryOp::Reverse.trait_fn();

            let left_comp_ty = left_comp.trait_ty();
            let right_comp_ty = right_comp.trait_ty();

            let predicates = &mut generics.make_where_clause().predicates;
            predicates.push(parse_quote!(T: #left_comp_ty<Output = U>));
            predicates.push(parse_quote!(U: #right_comp_ty<Output = T> + #rev_ty<Output = U>));

            let left_comp_fn = left_comp.trait_fn();
            let right_comp_fn = right_comp.trait_fn();
            let expr = quote!(self.#left_comp_fn().#rev_fn().#right_comp_fn());

            let (ig, _, wc) = generics.split_for_impl();

            Some(parse_quote! {
                impl #ig #trait_ for T #wc {
                    type Output = Self;
                    fn #fn_(self) -> Self::Output {
                        #expr
                    }
                }
            })
        } else {
            None
        }
    }

    pub fn impl_item(self, type_mv: AlgebraType) -> Vec<ItemImpl> {
        // traits with blanket impls do not need implementation
        if self.blanket_impl(type_mv.algebra()).is_some() {
            return vec![];
        }

        // cannot have blanket impls for foreign traits
        if matches!(self, Self::Neg) && type_mv.is_scalar() {
            return vec![];
        }

        vec![ItemImpl {
            attrs: vec![],
            defaultness: None,
            unsafety: None,
            impl_token: Impl::default(),
            generics: self.generics(type_mv),
            trait_: Some((None, self.trait_ty().convert(), For::default())),
            self_ty: Box::new(type_mv.ty_with_suffix("", None)),
            brace_token: Brace::default(),
            items: {
                let output_type = self.type_item(type_mv, None);
                let trait_fn = self.fn_item(type_mv);
                vec![output_type, trait_fn]
            },
        }]
    }

    fn output(self, ty: AlgebraType) -> AlgebraType {
        if ty.is_generic() {
            ty
        } else {
            AlgebraType::from_iter(self.products(ty), ty.algebra())
        }
    }

    fn generics(self, ty: AlgebraType) -> syn::Generics {
        let mut generics = Generics::default();

        if ty.has_float_generic(None) {
            generics.params.push(quote!(T).convert());
            generics
                .make_where_clause()
                .predicates
                .push(parse_quote!(T: num_traits::Float));
        }

        if !ty.is_generic() {
            return generics;
        }

        let output = self.output(ty);

        let input_params = ty.generics("").map::<GenericParam, _>(|ty| ty.convert());
        generics.params.extend(input_params);

        let output_params = output
            .generics(OUT_SUFFIX)
            .map::<GenericParam, _>(|ty| ty.convert());
        generics.params.extend(output_params);

        let trait_ty = self.trait_ty();
        let predicates = ty.algebra().grades().map::<WherePredicate, _>(|grade_in| {
            let grade_out = self.grade_out(grade_in);
            let out = grade_out.generic(OUT_SUFFIX);
            let ident = grade_in.generic("");
            parse_quote! {
                #ident: #trait_ty<Output = #out>
            }
        });
        let where_clause = generics.make_where_clause();
        where_clause.predicates.extend(predicates);

        generics
    }

    fn type_item(self, ty: AlgebraType, float: Option<FloatType>) -> ImplItem {
        let output = self.output(ty);
        let output = output.ty_with_suffix(OUT_SUFFIX, float);
        parse_quote! {
            type Output = #output;
        }
    }

    fn fn_item(self, ty: AlgebraType) -> ImplItem {
        let trait_fn = self.trait_fn();
        let expr = self.expr(ty);
        parse_quote! {
            #[inline]
            fn #trait_fn(self) -> Self::Output {
                #expr
            }
        }
    }

    fn expr(self, ty: AlgebraType) -> Expr {
        let output = self.output(ty);
        let trait_ty = self.trait_ty();
        let trait_fn = self.trait_fn();
        match output {
            AlgebraType::Zero(_) => Zero::expr(),
            AlgebraType::Grade(grade) => {
                let fields = grade.blades().map(|blade| {
                    let sum = ty
                        .blades()
                        .filter_map(|b| {
                            let product = self.call(b);
                            match product {
                                Product::Pos(b_out) if b_out == blade => {
                                    Some(access_blade(ty, b, quote!(self)))
                                }
                                Product::Neg(b_out) if b_out == blade => {
                                    let expr = access_blade(ty, b, quote!(self));
                                    Some(parse_quote! { - #expr })
                                }
                                _ => None,
                            }
                        })
                        .collect();
                    assign_blade(blade, &sum)
                });

                if let Some(ident) = grade.ident() {
                    parse_quote! {
                        #ident {
                            #( #fields, )*
                        }
                    }
                } else {
                    parse_quote!(#(#fields)*)
                }
            }
            AlgebraType::Multivector(mv) => {
                let ident = Multivector::ident();
                let fields = mv.1.grades().map(|grade_out| {
                    let exprs = mv.grades().filter_map(|grade_in| {
                        if self.grade_out(grade_in) == grade_out {
                            let f = access_grade(ty, grade_in, quote!(self));
                            Some(quote! { #trait_ty::#trait_fn(#f) })
                        } else {
                            None
                        }
                    });
                    quote!( #(#exprs)+* )
                });
                parse_quote! {
                    #ident(#(#fields),*)
                }
            }
        }
    }

    fn grade_out(self, grade: Grade) -> Grade {
        match self {
            Self::RightComplement | Self::LeftComplement | Self::Dual => {
                let out = grade.1.dimensions() - grade.0;
                grade.1.grade(out)
            }
            _ => grade,
        }
    }
}

impl SumOp {
    pub fn define(self) -> Option<ItemTrait> {
        if matches!(self, SumOp::Add | SumOp::Sub) {
            return None;
        }

        let ident = self.trait_ty();
        let trait_fn = self.trait_fn();
        Some(parse_quote! {
            pub trait #ident<Rhs> {
                type Output;
                fn #trait_fn(self, rhs: Rhs) -> Self::Output;
            }
        })
    }

    pub fn item_impl(self, lhs: AlgebraType, rhs: AlgebraType) -> Vec<ItemImpl> {
        if !self.is_local() && lhs.is_scalar() && rhs.is_scalar() {
            return vec![];
        }

        if self.is_grade_op() && (lhs.is_mv() || rhs.is_mv()) {
            return vec![];
        }

        if (!self.is_local() && lhs.is_scalar()) || (lhs.is_zero() && rhs.is_scalar()) {
            FloatType::iter()
                .map(|float| ItemImpl {
                    attrs: vec![],
                    defaultness: None,
                    unsafety: None,
                    impl_token: Impl::default(),
                    generics: self.generics(lhs, rhs, Some(float)),
                    trait_: {
                        let trait_ty = self.trait_ty();
                        let rhs_ty = rhs.ty_with_suffix(RHS_SUFFIX, Some(float));
                        Some((None, parse_quote!(#trait_ty<#rhs_ty>), For::default()))
                    },
                    self_ty: Box::new(lhs.ty_with_suffix_and_float(LHS_SUFFIX, Some(float))),
                    brace_token: Brace::default(),
                    items: {
                        let output_type = self.output_type_item(lhs, rhs, Some(float));
                        let trait_fn = self.fn_item(lhs, rhs, Some(float));
                        vec![output_type, trait_fn]
                    },
                })
                .collect()
        } else {
            vec![ItemImpl {
                attrs: vec![],
                defaultness: None,
                unsafety: None,
                impl_token: Impl::default(),
                generics: self.generics(lhs, rhs, None),
                trait_: {
                    let trait_ty = self.trait_ty();
                    let rhs_ty = rhs.ty_with_suffix(RHS_SUFFIX, None);
                    Some((None, parse_quote!(#trait_ty<#rhs_ty>), For::default()))
                },
                self_ty: Box::new(lhs.ty_with_suffix(LHS_SUFFIX, None)),
                brace_token: Brace::default(),
                items: {
                    let output_type = self.output_type_item(lhs, rhs, None);
                    let trait_fn = self.fn_item(lhs, rhs, None);
                    vec![output_type, trait_fn]
                },
            }]
        }
    }

    fn output(self, lhs: AlgebraType, rhs: AlgebraType) -> AlgebraType {
        AlgebraType::from_iter(self.products(lhs, rhs), lhs.algebra())
    }

    fn generics(self, lhs: AlgebraType, rhs: AlgebraType, float: Option<FloatType>) -> Generics {
        let trait_ty = self.trait_ty_grade();

        let mut generics = Generics::default();

        if lhs.has_float_generic(float) || rhs.has_float_generic(float) {
            generics.params.push(quote!(T).convert());
            generics
                .make_where_clause()
                .predicates
                .push(parse_quote!(T: num_traits::Float));
        }

        if lhs.is_generic() {
            let params = lhs
                .generics(LHS_SUFFIX)
                .map(|i| i.convert::<GenericParam>());
            generics.params.extend(params);
        }

        if rhs.is_generic() {
            let params = rhs
                .generics(RHS_SUFFIX)
                .map(|i| i.convert::<GenericParam>());
            generics.params.extend(params);
        }

        if is_generic(lhs, rhs) {
            let params = lhs
                .algebra()
                .grades()
                .filter_map(|grade| self.new_output_generic(grade, lhs, rhs))
                .map(|i| i.convert::<GenericParam>());
            generics.params.extend(params);

            let predicates = lhs
                .algebra()
                .grades()
                .filter_map::<WherePredicate, _>(|grade| {
                    if self.is_sub() && !lhs.contains(grade) {
                        let neg_ty = UnaryOp::Neg.trait_ty();
                        let out = self.new_output_generic(grade, lhs, rhs)?;
                        let rhs = grade_type(rhs, grade, RHS_SUFFIX, float);
                        Some(parse_quote!(#rhs: #neg_ty<Output = #out>))
                    } else {
                        let out = self.new_output_generic(grade, lhs, rhs)?;
                        let lhs = grade_type(lhs, grade, LHS_SUFFIX, float);
                        let rhs = grade_type(rhs, grade, RHS_SUFFIX, float);
                        Some(parse_quote!(#lhs: #trait_ty<#rhs, Output = #out>))
                    }
                });
            let where_clause = generics.make_where_clause();
            where_clause.predicates.extend(predicates);
        }

        generics
    }

    fn output_type_item(
        self,
        lhs: AlgebraType,
        rhs: AlgebraType,
        float: Option<FloatType>,
    ) -> ImplItem {
        let generic = is_generic(lhs, rhs);
        let output_ty = match self.output(lhs, rhs) {
            AlgebraType::Zero(_) => Zero::ty(),
            AlgebraType::Grade(grade) => {
                if generic {
                    unreachable!("mv +/- x != grade")
                } else {
                    grade.ty_with_float(float)
                }
            }
            AlgebraType::Multivector(mv) => {
                let ty = Multivector::ident();
                if generic {
                    let types = lhs
                        .algebra()
                        .grades()
                        .map(|grade| self.output_generic(grade, lhs, rhs));

                    parse_quote!( #ty <#(#types),*> )
                } else {
                    let types = mv.type_parameters("Out", float);
                    parse_quote!( #ty <#(#types),*> )
                }
            }
        };
        parse_quote! { type Output = #output_ty; }
    }

    fn output_generic(self, grade: Grade, lhs: AlgebraType, rhs: AlgebraType) -> Ident {
        match (lhs.contains(grade), rhs.contains(grade)) {
            (true, true) => grade.generic(OUT_SUFFIX),
            (true, false) => grade.generic(LHS_SUFFIX),
            (false, true) => {
                if self.is_sub() {
                    grade.generic(OUT_SUFFIX)
                } else {
                    grade.generic(RHS_SUFFIX)
                }
            }
            (false, false) => Zero::ident(),
        }
    }

    fn new_output_generic(self, grade: Grade, lhs: AlgebraType, rhs: AlgebraType) -> Option<Ident> {
        match (lhs.contains(grade), rhs.contains(grade)) {
            (true, true) => Some(grade.generic(OUT_SUFFIX)),
            (false, true) if self.is_sub() => Some(grade.generic(OUT_SUFFIX)),
            _ => None,
        }
    }

    fn fn_item(self, lhs: AlgebraType, rhs: AlgebraType, float: Option<FloatType>) -> ImplItem {
        let op_fn = self.trait_fn();
        let rhs_ty = rhs.ty_with_suffix(RHS_SUFFIX, float);
        let output_expr = self.sum_output_expr(lhs, rhs);
        parse_quote! {
            #[inline]
            #[allow(unused_variables)]
            fn #op_fn(self, rhs: #rhs_ty) -> Self::Output {
                #output_expr
            }
        }
    }

    fn sum_output_expr(self, lhs: AlgebraType, rhs: AlgebraType) -> Expr {
        let output = self.output(lhs, rhs);

        if output.is_zero() {
            return Zero::expr();
        }

        if rhs.is_zero() {
            return parse_quote!(self);
        }

        if lhs.is_zero() {
            return match self {
                SumOp::Add | SumOp::GradeAdd => parse_quote!(rhs),
                SumOp::Sub | SumOp::GradeSub => parse_quote!(-rhs),
            };
        }

        match self.output(lhs, rhs) {
            AlgebraType::Zero(_) => Zero::expr(),
            AlgebraType::Grade(grade) => {
                if is_generic(lhs, rhs) {
                    self.sum_grade_sum_expr(lhs, rhs, grade)
                } else {
                    sum_grade_fields_expr(self, grade, lhs, rhs)
                }
            }
            AlgebraType::Multivector(_) => {
                let ty = Multivector::ident();
                let grades = lhs
                    .algebra()
                    .grades()
                    .map(|grade| self.sum_grade_sum_expr(lhs, rhs, grade));
                parse_quote! { #ty( #(#grades),* ) }
            }
        }
    }

    fn sum_grade_sum_expr(self, lhs: AlgebraType, rhs: AlgebraType, grade: Grade) -> Expr {
        let lhs_expr = access_grade(lhs, grade, quote!(self));
        let rhs_expr = access_grade(rhs, grade, quote!(rhs));

        let trait_ty = self.trait_ty_grade();
        let trait_fn = self.trait_fn_grade();

        match (lhs.contains(grade), rhs.contains(grade)) {
            (true, true) => {
                parse_quote!(#trait_ty::#trait_fn(#lhs_expr, #rhs_expr))
            }
            (false, true) => match self {
                SumOp::Add | SumOp::GradeAdd => rhs_expr,
                SumOp::Sub | SumOp::GradeSub => parse_quote!(- #rhs_expr),
            },
            (true, false) => lhs_expr,
            (false, false) => Zero::expr(),
        }
    }
}

impl ProductOp {
    pub fn define(self) -> Option<syn::ItemTrait> {
        if matches!(self, ProductOp::Mul | ProductOp::Div) {
            return None;
        }

        let ident = self.trait_ty();
        let trait_fn = self.trait_fn();

        Some(parse_quote! {
            pub trait #ident<Rhs> {
                type Output;
                fn #trait_fn(self, rhs: Rhs) -> Self::Output;
            }
        })
    }

    pub fn blanket_impl(self, algebra: Algebra) -> Option<syn::ItemImpl> {
        match self {
            Self::Antigeometric | Self::Antidot | Self::Antiwedge => {
                let antitrait = self.trait_ty();
                let antifn = self.trait_fn();
                let inner_trait = match self {
                    Self::Antigeometric => Self::Geometric,
                    Self::Antidot => Self::Dot,
                    Self::Antiwedge => Self::Wedge,
                    _ => unreachable!(),
                };
                let trait_ = inner_trait.trait_ty();
                let trait_fn = inner_trait.trait_fn();
                let left_comp_ty = UnaryOp::left_comp(algebra).trait_ty();
                let left_comp_fn = UnaryOp::left_comp(algebra).trait_fn();
                let right_comp_ty = UnaryOp::right_comp(algebra).trait_ty();
                let right_comp_fn = UnaryOp::right_comp(algebra).trait_fn();

                Some(parse_quote! {
                    impl<Lhs, Rhs, LhsComp, RhsComp, OutputComp> #antitrait<Rhs> for Lhs
                    where
                        Lhs: #left_comp_ty<Output = LhsComp>,
                        Rhs: #left_comp_ty<Output = RhsComp>,
                        LhsComp: #trait_<RhsComp, Output = OutputComp>,
                        OutputComp: #right_comp_ty,
                    {
                        type Output = OutputComp::Output;
                        #[inline]
                        fn #antifn(self, rhs: Rhs) -> Self::Output {
                            let lhs = self.#left_comp_fn();
                            let rhs = rhs.#left_comp_fn();
                            let output_complement = lhs.#trait_fn(rhs);
                            output_complement.#right_comp_fn()
                        }
                    }
                })
            }
            _ => None,
        }
    }

    pub fn item_impl(self, lhs: AlgebraType, rhs: AlgebraType) -> Vec<ItemImpl> {
        // already covered by blanket impl
        if self.blanket_impl(lhs.algebra()).is_some() {
            return vec![];
        }

        // f32 and f64 already impl Mul and Div
        if lhs.is_scalar() && rhs.is_scalar() && !self.is_local() {
            return vec![];
        }

        // we'd need LeftDiv and RightDiv to do this properly for anything with grade > 0
        if matches!(self, ProductOp::Div) && !rhs.is_scalar() {
            return vec![];
        }

        if lhs.is_scalar() {
            FloatType::iter()
                .map(|float| ItemImpl {
                    attrs: vec![],
                    defaultness: None,
                    unsafety: None,
                    impl_token: Impl::default(),
                    generics: self.generics(lhs, rhs, Some(float)),
                    trait_: {
                        let trait_ty = self.trait_ty();
                        let rhs_ty = rhs.ty_with_suffix_and_float(RHS_SUFFIX, Some(float));
                        Some((None, parse_quote!(#trait_ty<#rhs_ty>), For::default()))
                    },
                    self_ty: Box::new(lhs.ty_with_suffix_and_float(LHS_SUFFIX, Some(float))),
                    brace_token: Brace::default(),
                    items: {
                        let output_type_item = self.output_type_item(lhs, rhs, Some(float));
                        let fn_item = self.fn_item(lhs, rhs, Some(float));
                        vec![output_type_item, fn_item]
                    },
                })
                .collect()
        } else {
            vec![ItemImpl {
                attrs: vec![],
                defaultness: None,
                unsafety: None,
                impl_token: Impl::default(),
                generics: self.generics(lhs, rhs, None),
                trait_: {
                    let trait_ty = self.trait_ty();
                    let rhs_ty = rhs.ty_with_suffix(RHS_SUFFIX, None);
                    Some((None, parse_quote!(#trait_ty<#rhs_ty>), For::default()))
                },
                self_ty: Box::new(lhs.ty_with_suffix(LHS_SUFFIX, None)),
                brace_token: Brace::default(),
                items: {
                    let output_type_item = self.output_type_item(lhs, rhs, None);
                    let fn_item = self.fn_item(lhs, rhs, None);
                    vec![output_type_item, fn_item]
                },
            }]
        }
    }

    fn generics(self, lhs: AlgebraType, rhs: AlgebraType, float: Option<FloatType>) -> Generics {
        let algebra = lhs.algebra();

        let mut generics = Generics::default();
        let mut copy_predicates = Vec::<WherePredicate>::new();
        let mut product_predicates = Vec::<WherePredicate>::new();
        let mut sum_predicates = Vec::<WherePredicate>::new();

        if lhs.has_float_generic(float) || rhs.has_float_generic(float) {
            generics.params.push(parse_quote!(T));
            generics
                .make_where_clause()
                .predicates
                .push(parse_quote!(T: num_traits::Float));
        }

        if !is_generic(lhs, rhs) {
            return generics;
        }

        for grade in lhs.generics(LHS_SUFFIX).chain(rhs.generics(RHS_SUFFIX)) {
            generics.params.push(grade.convert());
            let predicate = parse_quote! { #grade: Copy };
            copy_predicates.push(predicate);
        }

        for grade in algebra.grades() {
            let int = self.intermediate_types(lhs, rhs, grade);
            let sum = self.intermediate_and_sum_types(lhs, rhs, grade);

            for ident in (0..sum).into_iter().map(|n| grade.generic_n(n)) {
                generics.params.push(ident.convert());
            }

            let inner_op = self.grade_op(grade);

            let op_trait = inner_op.trait_ty();
            for (n, (lhs_grade, rhs_grade)) in iproduct!(lhs.grades(), rhs.grades())
                .filter(|(lhs, rhs)| self.output_contains(*lhs, *rhs, grade))
                .enumerate()
            {
                let lhs = grade_type(lhs, lhs_grade, LHS_SUFFIX, float);
                let rhs = grade_type(rhs, rhs_grade, RHS_SUFFIX, float);
                let out = grade.generic_n(n);

                let predicate = parse_quote!( #lhs: #op_trait<#rhs, Output = #out> );
                product_predicates.push(predicate);
            }

            let lhs = once(0).chain(int..sum).map(|n| grade.generic_n(n));
            let rhs = (1..int).into_iter().map(|n| grade.generic_n(n));
            let out = (int..sum).into_iter().map(|n| grade.generic_n(n));
            for ((lhs, rhs), out) in lhs.zip(rhs).zip(out) {
                let predicate = parse_quote! { #lhs: std::ops::Add<#rhs, Output = #out> };
                sum_predicates.push(predicate);
            }
        }

        let predicates = &mut generics.make_where_clause().predicates;
        predicates.extend(product_predicates);
        predicates.extend(copy_predicates);
        predicates.extend(sum_predicates);

        generics
    }

    fn last_generics(self, lhs: AlgebraType, rhs: AlgebraType) -> impl Iterator<Item = Ident> {
        lhs.algebra()
            .grades()
            .map(move |grade| self.last_generic(lhs, rhs, grade))
    }

    fn last_generic(self, lhs: AlgebraType, rhs: AlgebraType, grade: Grade) -> Ident {
        self.intermediate_and_sum_types(lhs, rhs, grade)
            .checked_sub(1)
            .map(|n| grade.generic_n(n))
            .unwrap_or_else(Zero::ident)
    }

    fn intermediate_and_sum_types(self, lhs: AlgebraType, rhs: AlgebraType, grade: Grade) -> usize {
        use std::ops::Mul;
        self.intermediate_types(lhs, rhs, grade)
            .mul(2)
            .checked_sub(1)
            .unwrap_or_default()
    }

    fn intermediate_types(self, lhs: AlgebraType, rhs: AlgebraType, grade: Grade) -> usize {
        iproduct!(lhs.grades(), rhs.grades())
            .filter(|(lhs, rhs)| self.output_contains(*lhs, *rhs, grade))
            .count()
    }

    fn output_type_item(
        self,
        lhs: AlgebraType,
        rhs: AlgebraType,
        float: Option<FloatType>,
    ) -> ImplItem {
        let is_generic = is_generic(lhs, rhs);
        let output_ty = match self.output(lhs, rhs) {
            AlgebraType::Zero(_) => Zero::ty(),
            AlgebraType::Grade(grade) => {
                if is_generic {
                    self.last_generic(lhs, rhs, grade).convert()
                } else {
                    grade.ty_with_float(float)
                }
            }
            AlgebraType::Multivector(mv) => {
                let ty = Multivector::ident();
                if is_generic {
                    let types = self.last_generics(lhs, rhs);
                    parse_quote!( #ty <#(#types),*> )
                } else {
                    let types = mv.type_parameters(OUT_SUFFIX, float);
                    parse_quote!( #ty <#(#types),*> )
                }
            }
        };
        parse_quote! { type Output = #output_ty; }
    }

    fn fn_item(self, lhs: AlgebraType, rhs: AlgebraType, float: Option<FloatType>) -> ImplItem {
        let op_fn = self.trait_fn();
        let rhs_ty = rhs.ty_with_suffix_and_float(RHS_SUFFIX, float);
        let output_expr = self.output_expr(lhs, rhs);
        parse_quote! {
            #[inline]
            #[allow(unused_variables)]
            fn #op_fn(self, rhs: #rhs_ty) -> Self::Output {
                #output_expr
            }
        }
    }

    fn output_expr(self, lhs: AlgebraType, rhs: AlgebraType) -> Expr {
        match self.output(lhs, rhs) {
            AlgebraType::Zero(_) => Zero::expr(),
            AlgebraType::Grade(grade) => {
                if is_generic(lhs, rhs) {
                    let products = iproduct!(lhs.grades(), rhs.grades()).filter_map::<Expr, _>(
                        |(lhs_grade, rhs_grade)| {
                            self.grade_product_expr(grade, lhs, lhs_grade, rhs, rhs_grade)
                        },
                    );
                    parse_quote! { #(#products)+* }
                } else {
                    self.grade_fields_expr(grade, lhs, rhs)
                }
            }
            AlgebraType::Multivector(mv) => {
                let ident = Multivector::ident();
                let fields = mv.1.grades().map(|grade| {
                    let sum = iproduct!(lhs.grades(), rhs.grades())
                        .filter_map(|(lhs_grade, rhs_grade)| {
                            self.grade_product_expr(grade, lhs, lhs_grade, rhs, rhs_grade)
                        })
                        .collect::<Vec<_>>();
                    if sum.is_empty() {
                        Zero::expr()
                    } else {
                        parse_quote! { #( #sum )+* }
                    }
                });
                parse_quote! { #ident( #(#fields),* ) }
            }
        }
    }

    pub fn grade_product_expr(
        self,
        grade: Grade,
        lhs: AlgebraType,
        lhs_grade: Grade,
        rhs: AlgebraType,
        rhs_grade: Grade,
    ) -> Option<Expr> {
        if self.output_contains(lhs_grade, rhs_grade, grade) {
            let op_fn = self.grade_op(grade).trait_fn();
            let lhs = access_grade(lhs, lhs_grade, quote!(self));
            let rhs = access_grade(rhs, rhs_grade, quote!(rhs));
            Some(parse_quote! { #lhs.#op_fn(#rhs) })
        } else {
            None
        }
    }

    fn grade_fields_expr(self, grade: Grade, lhs: AlgebraType, rhs: AlgebraType) -> syn::Expr {
        let fields = grade.blades().map(|blade| {
            let sum = cartesian_product(lhs, rhs, self)
                .filter_map(|(lhs_blade, rhs_blade, _)| {
                    self.product_expr(lhs, lhs_blade, rhs, rhs_blade, blade)
                })
                .collect();
            assign_blade(blade, &sum)
        });

        if let Some(ident) = grade.ident() {
            parse_quote! {
                #ident {
                    #( #fields, )*
                }
            }
        } else {
            parse_quote!(#(#fields)*)
        }
    }
}

impl NormOps {
    pub fn define_and_blanket(self) -> TokenStream {
        match self {
            Self::Norm2 => quote! {
                pub trait Norm2 {
                    type Output;
                    fn norm2(self) -> Self::Output;
                }

                impl<T, S> Norm2 for T
                where
                    T: ScalarProduct<T, Output = S> + Reverse<Output = T> + Copy,
                {
                    type Output = S;
                    #[inline]
                    fn norm2(self) -> Self::Output {
                        self.scalar_prod(self.rev())
                    }
                }
            },
            Self::Norm => quote! {
                pub trait Norm {
                    type Output;
                    fn norm(self) -> Self::Output;
                }

                impl<T, S> Norm for T
                where
                    T: Norm2<Output = S>,
                    S: num_traits::Float,
                {
                    type Output = S;
                    #[inline]
                    fn norm(self) -> Self::Output {
                        self.norm2().sqrt()
                    }
                }
            },
            Self::Inverse => quote! {
                pub trait Inverse {
                    fn inv(self) -> Self;
                }

                impl<T, S> Inverse for T
                where
                    T: Reverse<Output = T> + Norm2<Output = S> + std::ops::Div<S, Output = T> + Copy,
                    S: num_traits::Float,
                {
                    #[inline]
                    fn inv(self) -> Self {
                        let norm2 = self.norm2();
                        if norm2 == S::zero() {
                            panic!("div by zero");
                        }
                        self.rev() / norm2
                    }
                }
            },
            Self::Unit => quote! {
                pub trait Unit {
                    fn unit(self) -> Self;
                }

                impl<T, S> Unit for T
                where
                    T: Reverse<Output = T> + Norm<Output = S> + std::ops::Div<S, Output = T> + Copy,
                    S: num_traits::Float,
                {
                    #[inline]
                    fn unit(self) -> Self {
                        let norm = self.norm();
                        if norm == S::zero() {
                            panic!("div by zero");
                        }
                        self.rev() / norm
                    }
                }
            },
        }
    }
}

fn is_generic(lhs: AlgebraType, rhs: AlgebraType) -> bool {
    lhs.is_generic() || rhs.is_generic()
}

fn sum_grade_fields_expr(op: SumOp, grade: Grade, lhs: AlgebraType, rhs: AlgebraType) -> syn::Expr {
    let trait_ty = op.trait_ty_std();
    let trait_fn = op.trait_fn_std();

    let fields = grade.blades().map(|blade| {
        let lhs = access_blade(lhs, blade, quote!(self));
        let rhs = access_blade(rhs, blade, quote!(rhs));
        quote!(#trait_ty::#trait_fn(#lhs, #rhs))
    });

    if let Some(ident) = grade.ident() {
        let idents = grade.blades().map(|blade| {
            let field = blade.field().unwrap();
            quote!(#field)
        });

        parse_quote! {
            #ident {
                #(#idents: #fields,)*
            }
        }
    } else {
        parse_quote!(#(#fields)*)
    }
}

const LHS_SUFFIX: &str = "Lhs";
const RHS_SUFFIX: &str = "Rhs";
const OUT_SUFFIX: &str = "Out";

fn grade_type(ty: AlgebraType, grade: Grade, suffix: &str, float: Option<FloatType>) -> syn::Type {
    if ty.is_generic() {
        grade.generic(suffix).convert()
    } else if ty.is_scalar() {
        ty.ty_with_suffix_and_float(suffix, float)
    } else {
        grade.ty_with_float(float)
    }
}

fn access_grade(parent: AlgebraType, grade: Grade, ident: TokenStream) -> syn::Expr {
    let field = match parent {
        AlgebraType::Multivector(_) => grade.mv_field(),
        AlgebraType::Zero(_) => unreachable!("no grades to access"),
        AlgebraType::Grade(_) => return ident.convert(),
    };
    parse_quote! {
        #ident.#field
    }
}

fn assign_blade(blade: Blade, sum: &Vec<syn::Expr>) -> TokenStream {
    let expr = if sum.is_empty() {
        quote! { T::zero() }
    } else {
        quote! { #( #sum )+* }
    };
    if let Some(field) = blade.field() {
        quote! { #field: #expr }
    } else {
        quote! { #expr }
    }
}

impl AlgebraType {
    fn define(self) -> Option<ItemStruct> {
        match self {
            Self::Zero(_) => Some(parse_quote! {
                #[repr(C)]
                #[derive(Debug, Default, Copy, Clone, PartialEq)]
                pub struct Zero;
            }),
            Self::Grade(grade) => {
                grade.ident()?;

                let ty = grade.ty();
                let fields = grade.blades().map(|blade| {
                    let f = blade.field().unwrap();
                    quote! { pub #f: T, }
                });
                Some(parse_quote! {
                    #[repr(C)]
                    #[derive(Debug, Default, Copy, Clone, PartialEq)]
                    pub struct #ty {
                        #(#fields)*
                    }
                })
            }
            Self::Multivector(mv) => {
                let generics = mv.type_parameters("", None);

                let fields = generics.iter().map(|g| {
                    quote! {
                        pub #g,
                    }
                });

                Some(parse_quote! {
                    #[repr(C)]
                    #[derive(Debug, Default, Copy, Clone, PartialEq)]
                    pub struct Multivector < #(#generics),* > ( #(#fields)* );
                })
            }
        }
    }

    pub fn ty_with_suffix(&self, suffix: &str, float: Option<FloatType>) -> syn::Type {
        match self {
            Self::Zero(_) => Zero::ty(),
            Self::Grade(g) => g.ty_with_float(float),
            Self::Multivector(mv) => {
                let ty = Multivector::ident();
                let generics = mv.type_parameters(suffix, float);
                parse_quote! {
                    #ty < #(#generics),* >
                }
            }
        }
    }

    pub fn ty_with_suffix_and_float(&self, suffix: &str, float: Option<FloatType>) -> syn::Type {
        match self {
            Self::Zero(_) => Zero::ty(),
            Self::Grade(g) => g.ty_with_float(float),
            Self::Multivector(mv) => {
                let ty = Multivector::ident();
                let generics = mv.type_parameters(suffix, float);
                parse_quote! {
                    #ty < #(#generics),* >
                }
            }
        }
    }

    fn impl_item(self) -> Option<ItemImpl> {
        match self {
            Self::Grade(_) => {
                let item = self.new_fn()?;
                let ty = self.ty_with_suffix("", None);
                Some::<ItemImpl>(parse_quote! {
                    impl<T> #ty {
                        #item
                    }
                })
            }
            _ => None,
        }
    }

    fn new_fn(self) -> Option<ImplItem> {
        match self {
            AlgebraType::Grade(grade) => {
                let ty = grade.ty();
                let ident = grade.ident()?;
                let fields = grade.blades().map(|b| {
                    let f = b.field().unwrap();
                    quote! { #f: T }
                });
                let expr_fields = grade.blades().map(|b| b.field().unwrap());
                Some(parse_quote! {
                    #[inline]
                    pub const fn new(#(#fields),*) -> #ty {
                        #ident {
                            #(#expr_fields,)*
                        }
                    }
                })
            }
            _ => None,
        }
    }
}

impl FloatConvert {
    pub fn define(self) -> ItemTrait {
        let trait_ = self.trait_ty();
        let fn_ = self.trait_fn();
        parse_quote! {
            pub trait #trait_ {
                type Output;
                fn #fn_(self) -> Self::Output;
            }
        }
    }

    fn trait_ty(self) -> syn::Type {
        match self {
            Self::F32 => parse_quote!(ToF32),
            Self::F64 => parse_quote!(ToF64),
        }
    }

    fn trait_fn(self) -> Ident {
        match self {
            Self::F32 => parse_quote!(to_f32),
            Self::F64 => parse_quote!(to_f64),
        }
    }

    fn input_float(self) -> FloatType {
        match self {
            Self::F32 => FloatType::F64,
            Self::F64 => FloatType::F32,
        }
    }

    fn output_float(self) -> FloatType {
        match self {
            Self::F32 => FloatType::F32,
            Self::F64 => FloatType::F64,
        }
    }

    fn impl_item(self, ty: AlgebraType) -> ItemImpl {
        let trait_ = self.trait_ty();
        let fn_ = self.trait_fn();
        match ty {
            AlgebraType::Zero(_) => parse_quote! {
                impl #trait_ for Zero {
                    type Output = Zero;
                    fn #fn_(self) -> Self::Output {
                        Zero
                    }
                }
            },
            AlgebraType::Grade(grade) if grade.is_scalar() => {
                let input = self.input_float().ty();
                let output = self.output_float().ty();
                parse_quote! {
                    impl #trait_ for #input {
                        type Output = #output;
                        fn #fn_(self) -> Self::Output {
                            self as #output
                        }
                    }
                }
            }
            AlgebraType::Grade(grade) => {
                let out_float = self.output_float().ty();
                let input = grade.ty_with_float(Some(self.input_float()));
                let output = grade.ty_with_float(Some(self.output_float()));
                let fields = grade.blades().map(|b| {
                    if let Some(field) = b.field() {
                        quote!(#field: self.#field as #out_float)
                    } else {
                        quote!(self as #out_float)
                    }
                });
                let expr = if let Some(ident) = grade.ident() {
                    parse_quote! {
                        #ident {
                            #(#fields),*
                        }
                    }
                } else {
                    quote!(#(#fields),*)
                };
                parse_quote! {
                    impl #trait_ for #input {
                        type Output = #output;
                        fn #fn_(self) -> Self::Output {
                            #expr
                        }
                    }
                }
            }
            AlgebraType::Multivector(mv) => {
                let input = ty.ty_with_suffix(LHS_SUFFIX, None);
                let output = ty.ty_with_suffix(OUT_SUFFIX, None);

                let mut generics = Generics::default();
                for (input, output) in mv
                    .type_parameters(LHS_SUFFIX, None)
                    .iter()
                    .zip(&mv.type_parameters(OUT_SUFFIX, None))
                {
                    generics.params.push(input.convert());
                    generics.params.push(output.convert());

                    generics
                        .make_where_clause()
                        .predicates
                        .push(parse_quote!(#input: #trait_<Output = #output>));
                }

                let fields = ty.algebra().grades().map(|grade| {
                    let g = grade.mv_field();
                    quote!(self.#g.#fn_())
                });

                let (ig, _, wc) = generics.split_for_impl();

                parse_quote! {
                    impl #ig #trait_ for #input #wc {
                        type Output = #output;
                        fn #fn_(self) -> Self::Output {
                            Multivector( #(#fields),* )
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn write_to_file(tokens: &impl ToTokens) {
        let contents = tokens.to_token_stream().to_string();
        let file = "../output.rs";
        std::fs::write(file, contents).unwrap();
    }

    #[track_caller]
    fn assert_eq_impl(expected: &ItemImpl, actual: &ItemImpl) {
        assert_eq_slice(&expected.attrs, &actual.attrs, "attrs");
        assert_eq_ref(&expected.defaultness, &actual.defaultness, "default");
        assert_eq_ref(&expected.unsafety, &actual.unsafety, "unsafety");
        assert_eq_ref(&expected.impl_token, &actual.impl_token, "impl_token");

        match (&expected.trait_, &actual.trait_) {
            (Some(te), Some(ta)) => {
                assert_eq_ref(&te.1, &ta.1, "trait");
            }
            (None, None) => {}
            _ => panic!("ItemImpl.trait_ are not eq"),
        }

        assert_eq_slice(&expected.items, &actual.items, "items");
        assert_eq_generics(&expected.generics, &actual.generics);

        assert_eq_ref(&expected, &actual, "full impl");
    }

    #[track_caller]
    fn assert_eq_ref<T: ToTokens>(expected: &T, actual: &T, message: &str) {
        assert_eq!(
            expected.to_token_stream().to_string(),
            actual.to_token_stream().to_string(),
            "{}",
            message
        );
    }

    #[track_caller]
    fn assert_eq_slice<T: ToTokens>(expected: &[T], actual: &[T], message: &str) {
        for (expected, actual) in expected.iter().zip(actual) {
            assert_eq_ref(expected, actual, message);
        }
    }

    #[track_caller]
    fn assert_eq_generics(expected: &Generics, actual: &Generics) {
        let (ea, eb, ec) = expected.split_for_impl();
        let (aa, ab, ac) = actual.split_for_impl();
        assert_eq_ref(&ea, &aa, "impl generics");
        assert_eq_ref(&eb, &ab, "type generics");

        assert_eq!(ec.is_some(), ac.is_some(), "where clause some/none");
        if let (Some(ec), Some(ac)) = (ec, ac) {
            assert_eq!(ec.predicates.len(), ac.predicates.len(), "predicate len");
            for (e, a) in ec.predicates.iter().zip(ac.predicates.iter()) {
                assert_eq!(
                    e.to_token_stream().to_string(),
                    a.to_token_stream().to_string(),
                    "predicate"
                );
            }
        }
    }

    #[test]
    fn mv_mv_bivector_product() {
        let algebra = Algebra::new(3, 0, 0);
        let mv = AlgebraType::Multivector(Multivector::new(algebra));

        let impl_item = &ProductOp::Grade(algebra.grade(2)).item_impl(mv, mv)[0];
        let expected: ItemImpl = parse_quote! {
            impl < G0Lhs , G1Lhs , G2Lhs , G3Lhs , G0Rhs , G1Rhs , G2Rhs , G3Rhs , G2_0 , G2_1 , G2_2 , G2_3 , G2_4 , G2_5 , G2_6 , G2_7 , G2_8 , G2_9 , G2_10 > BivectorProduct < Multivector < G0Rhs , G1Rhs , G2Rhs , G3Rhs > > for Multivector < G0Lhs , G1Lhs , G2Lhs , G3Lhs > where G0Lhs : BivectorProduct < G2Rhs , Output = G2_0 > , G1Lhs : BivectorProduct < G1Rhs , Output = G2_1 > , G1Lhs : BivectorProduct < G3Rhs , Output = G2_2 > , G2Lhs : BivectorProduct < G0Rhs , Output = G2_3 > , G2Lhs : BivectorProduct < G2Rhs , Output = G2_4 > , G3Lhs : BivectorProduct < G1Rhs , Output = G2_5 > , G0Lhs : Copy , G1Lhs : Copy , G2Lhs : Copy , G3Lhs : Copy , G0Rhs : Copy , G1Rhs : Copy , G2Rhs : Copy , G3Rhs : Copy , G2_0 : std :: ops :: Add < G2_1 , Output = G2_6 > , G2_6 : std :: ops :: Add < G2_2 , Output = G2_7 > , G2_7 : std :: ops :: Add < G2_3 , Output = G2_8 > , G2_8 : std :: ops :: Add < G2_4 , Output = G2_9 > , G2_9 : std :: ops :: Add < G2_5 , Output = G2_10 > { type Output = G2_10 ; # [inline] # [allow (unused_variables)] fn bivector_prod (self , rhs : Multivector < G0Rhs , G1Rhs , G2Rhs , G3Rhs >) -> Self :: Output { self . 0 . bivector_prod (rhs . 2) + self . 1 . bivector_prod (rhs . 1) + self . 1 . bivector_prod (rhs . 3) + self . 2 . bivector_prod (rhs . 0) + self . 2 . bivector_prod (rhs . 2) + self . 3 . bivector_prod (rhs . 1) } }
        };

        assert_eq_impl(&expected, impl_item);
    }

    #[test]
    fn mv_mv_product() {
        let algebra = Algebra::new(2, 0, 0);
        let mv = AlgebraType::Multivector(Multivector::new(algebra));

        let impl_item = &ProductOp::Mul.item_impl(mv, mv)[0];
        let expected = parse_quote! {
            impl < G0Lhs , G1Lhs , G2Lhs , G0Rhs , G1Rhs , G2Rhs , G0_0 , G0_1 , G0_2 , G0_3 , G0_4 , G1_0 , G1_1 , G1_2 , G1_3 , G1_4 , G1_5 , G1_6 , G2_0 , G2_1 , G2_2 , G2_3 , G2_4 > std :: ops :: Mul < Multivector < G0Rhs , G1Rhs , G2Rhs > > for Multivector < G0Lhs , G1Lhs , G2Lhs > where G0Lhs : ScalarProduct < G0Rhs , Output = G0_0 > , G1Lhs : ScalarProduct < G1Rhs , Output = G0_1 > , G2Lhs : ScalarProduct < G2Rhs , Output = G0_2 > , G0Lhs : VectorProduct < G1Rhs , Output = G1_0 > , G1Lhs : VectorProduct < G0Rhs , Output = G1_1 > , G1Lhs : VectorProduct < G2Rhs , Output = G1_2 > , G2Lhs : VectorProduct < G1Rhs , Output = G1_3 > , G0Lhs : BivectorProduct < G2Rhs , Output = G2_0 > , G1Lhs : BivectorProduct < G1Rhs , Output = G2_1 > , G2Lhs : BivectorProduct < G0Rhs , Output = G2_2 > , G0Lhs : Copy , G1Lhs : Copy , G2Lhs : Copy , G0Rhs : Copy , G1Rhs : Copy , G2Rhs : Copy , G0_0 : std :: ops :: Add < G0_1 , Output = G0_3 > , G0_3 : std :: ops :: Add < G0_2 , Output = G0_4 > , G1_0 : std :: ops :: Add < G1_1 , Output = G1_4 > , G1_4 : std :: ops :: Add < G1_2 , Output = G1_5 > , G1_5 : std :: ops :: Add < G1_3 , Output = G1_6 > , G2_0 : std :: ops :: Add < G2_1 , Output = G2_3 > , G2_3 : std :: ops :: Add < G2_2 , Output = G2_4 > { type Output = Multivector < G0_4 , G1_6 , G2_4 > ; # [inline] # [allow (unused_variables)] fn mul (self , rhs : Multivector < G0Rhs , G1Rhs , G2Rhs >) -> Self :: Output { Multivector (self . 0 . scalar_prod (rhs . 0) + self . 1 . scalar_prod (rhs . 1) + self . 2 . scalar_prod (rhs . 2) , self . 0 . vector_prod (rhs . 1) + self . 1 . vector_prod (rhs . 0) + self . 1 . vector_prod (rhs . 2) + self . 2 . vector_prod (rhs . 1) , self . 0 . bivector_prod (rhs . 2) + self . 1 . bivector_prod (rhs . 1) + self . 2 . bivector_prod (rhs . 0)) } }
        };

        assert_eq_impl(&expected, impl_item);
    }

    #[test]
    fn vector_product_of_two_vectors_is_zero() {
        let algebra = Algebra::new(2, 0, 0);
        let vector = AlgebraType::Grade(algebra.grade(1));

        // <vector * vector>_1 = 0
        let impl_item = &ProductOp::Grade(algebra.grade(1)).item_impl(vector, vector)[0];
        let expected = parse_quote! {
            impl<T> VectorProduct < Vector<T> > for Vector<T>
            where
                T: num_traits::Float
            {
                type Output = Zero ;
                # [inline]
                # [allow (unused_variables)]
                fn vector_prod (self , rhs : Vector<T>) -> Self :: Output { Zero }
            }
        };

        assert_eq_impl(&expected, impl_item);
    }

    #[test]
    fn vector_product_of_vectors_and_bivector_is_vector() {
        let algebra = Algebra::new(3, 0, 0);
        let vector = AlgebraType::Grade(algebra.grade(1));
        let bivector = AlgebraType::Grade(algebra.grade(2));

        // <vector * bivector>_1 = vector
        let impl_item = &ProductOp::Grade(algebra.grade(1)).item_impl(vector, bivector)[0];
        let expected = parse_quote! {
            impl<T> VectorProduct < Bivector<T> > for Vector<T>
            where
                T: num_traits::Float
            {
                type Output = Vector<T> ;
                # [inline]
                # [allow (unused_variables)]
                fn vector_prod (self , rhs : Bivector<T>) -> Self :: Output {
                    Vector {
                        e1 : - (self . e2 * rhs . e12) + - (self . e3 * rhs . e13) ,
                        e2 : self . e1 * rhs . e12 + - (self . e3 * rhs . e23) ,
                        e3 : self . e1 * rhs . e13 + self . e2 * rhs . e23 ,
                    }
                }
            }
        };

        assert_eq_impl(&expected, impl_item);
    }

    #[test]
    fn vector_vector_product() {
        let algebra = Algebra::new(3, 0, 0);
        let vector = AlgebraType::Grade(algebra.grade(1));

        let impl_item = &ProductOp::Mul.item_impl(vector, vector)[0];
        let expected = parse_quote! {
            impl<T> std::ops::Mul<Vector<T>> for Vector<T>
            where
                T: num_traits::Float
            {
                type Output = Multivector<T, Zero, Bivector<T>, Zero>;
                #[inline]
                #[allow(unused_variables)]
                fn mul(self, rhs: Vector<T>) -> Self::Output {
                    Multivector(self.scalar_prod(rhs), Zero, self.bivector_prod(rhs), Zero)
                }
            }
        };

        assert_eq_impl(&expected, impl_item);
    }

    #[test]
    fn scalar_scalar_geo() {
        let algebra = Algebra::new(3, 0, 0);
        let scalar = AlgebraType::Grade(algebra.grade(0));

        let impl_item = &ProductOp::Geometric.item_impl(scalar, scalar)[0];
        let expected = parse_quote! {
            impl Geometric<f32> for f32 {
                type Output = f32;
                #[inline]
                #[allow(unused_variables)]
                fn geo(self, rhs: f32) -> Self::Output {
                    self * rhs
                }
            }
        };

        assert_eq_impl(&expected, impl_item);
    }

    #[test]
    pub fn scalar_mv_scalar_product() {
        let g2 = Algebra::new(2, 0, 0);
        let scalar = g2.grade(0);

        let actual = &ProductOp::Grade(scalar).item_impl(AlgebraType::Grade(scalar), g2.mv())[0];

        let expected = parse_quote! {
            impl<G0Rhs, G1Rhs, G2Rhs, G0_0> ScalarProduct<Multivector<G0Rhs, G1Rhs, G2Rhs>> for f32
            where
                f32: ScalarProduct<G0Rhs, Output = G0_0>,
                G0Rhs: Copy,
                G1Rhs: Copy,
                G2Rhs: Copy
            {
                type Output = G0_0;
                #[inline]
                #[allow(unused_variables)]
                fn scalar_prod(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs>) -> Self::Output {
                    self.scalar_prod(rhs.0)
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    pub fn scalar_mv_mul() {
        let algebra = Algebra::new(3, 0, 0);
        let scalar = algebra.grade(0);

        let actual = ProductOp::Mul.item_impl(AlgebraType::Grade(scalar), algebra.mv());
        assert_eq!(2, actual.len());

        let actual = &actual[0];

        let expected = parse_quote! {
            impl<G0Rhs, G1Rhs, G2Rhs, G3Rhs, G0_0, G1_0, G2_0, G3_0>
                std::ops::Mul<Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>> for f32
            where
                f32: ScalarProduct<G0Rhs, Output = G0_0>,
                f32: VectorProduct<G1Rhs, Output = G1_0>,
                f32: BivectorProduct<G2Rhs, Output = G2_0>,
                f32: TrivectorProduct<G3Rhs, Output = G3_0>,
                G0Rhs: Copy,
                G1Rhs: Copy,
                G2Rhs: Copy,
                G3Rhs: Copy
            {
                type Output = Multivector<G0_0, G1_0, G2_0, G3_0>;
                #[inline]
                #[allow(unused_variables)]
                fn mul(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs, G3Rhs>) -> Self::Output {
                    Multivector(
                        self.scalar_prod(rhs.0),
                        self.vector_prod(rhs.1),
                        self.bivector_prod(rhs.2),
                        self.trivector_prod(rhs.3)
                    )
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn mul_scalars_is_none() {
        let algebra = Algebra::new(3, 0, 0);
        let scalar = AlgebraType::Grade(algebra.grade(0));
        let item_impl = ProductOp::Mul.item_impl(scalar, scalar);
        assert!(item_impl.is_empty());
    }

    #[test]
    fn intermediate_type_test() {
        let algebra = Algebra::new(3, 0, 0);
        let mv = algebra.mv();
        let bivector = algebra.grade(2);
        let trivector = algebra.grade(3);

        assert_eq!(4, ProductOp::Mul.intermediate_types(mv, mv, trivector));

        assert_eq!(
            1,
            ProductOp::Mul.intermediate_types(mv, AlgebraType::Grade(bivector), trivector)
        );

        assert_eq!(6, ProductOp::Mul.intermediate_types(mv, mv, bivector));
    }

    #[test]
    fn last_type_test() {
        let algebra = Algebra::new(3, 0, 0);
        let mv = algebra.mv();
        let bivector = algebra.grade(2);

        assert_eq_ref(
            &bivector.generic_n(10),
            &ProductOp::Mul.last_generic(mv, mv, bivector),
            "ident",
        );

        assert_eq_ref(
            &Zero::ident(),
            &ProductOp::Grade(algebra.grade(0)).last_generic(mv, mv, bivector),
            "ident",
        );
    }

    #[test]
    fn mv_generics_test() {
        let algebra = Algebra::new(2, 0, 0);
        let mv = algebra.mv();
        let op = ProductOp::Mul;
        let item_impl = &op.item_impl(mv, mv)[0];

        let expected = &item_impl.generics;
        let actual = op.generics(mv, mv, None);

        assert_eq!(
            expected.where_clause.as_ref().unwrap().predicates.len(),
            actual.where_clause.as_ref().unwrap().predicates.len()
        );

        assert_eq_generics(&expected, &actual);
    }

    #[test]
    fn sum_zeros() {
        let algebra = Algebra::new(3, 0, 0);
        let zero = AlgebraType::Zero(algebra);

        let actual = &SumOp::Add.item_impl(zero, zero)[0];
        let expected = parse_quote! {
            impl std::ops::Add<Zero> for Zero {
                type Output = Zero;
                #[inline]
                #[allow(unused_variables)]
                fn add(self, rhs: Zero) -> Self::Output {
                    Zero
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn sum_vectors() {
        let algebra = Algebra::new(3, 0, 0);
        let vector = AlgebraType::Grade(algebra.grade(1));

        let actual = &SumOp::Add.item_impl(vector, vector)[0];
        let expected = parse_quote! {
            impl<T> std::ops::Add<Vector<T>> for Vector<T>
            where
                T: num_traits::Float
            {
                type Output = Vector<T>;
                #[inline]
                #[allow(unused_variables)]
                fn add(self, rhs: Vector<T>) -> Self::Output {
                    Vector {
                        e1: std::ops::Add::add(self.e1, rhs.e1),
                        e2: std::ops::Add::add(self.e2, rhs.e2),
                        e3: std::ops::Add::add(self.e3, rhs.e3),
                    }
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn sub_vectors() {
        let algebra = Algebra::new(3, 0, 0);
        let vector = AlgebraType::Grade(algebra.grade(1));

        let actual = &SumOp::Sub.item_impl(vector, vector)[0];
        let expected = parse_quote! {
            impl<T> std::ops::Sub<Vector<T>> for Vector<T>
            where
                T: num_traits::Float
            {
                type Output = Vector<T>;
                #[inline]
                #[allow(unused_variables)]
                fn sub(self, rhs: Vector<T>) -> Self::Output {
                    Vector {
                        e1: std::ops::Sub::sub(self.e1, rhs.e1),
                        e2: std::ops::Sub::sub(self.e2, rhs.e2),
                        e3: std::ops::Sub::sub(self.e3, rhs.e3),
                    }
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn sum_vector_and_zero() {
        let algebra = Algebra::new(3, 0, 0);
        let zero = AlgebraType::Zero(algebra);
        let vector = AlgebraType::Grade(algebra.grade(1));

        let actual = &SumOp::Sub.item_impl(vector, zero)[0];
        let expected = parse_quote! {
            impl<T> std::ops::Sub<Zero> for Vector<T>
            where
                T: num_traits::Float
            {
                type Output = Vector<T>;
                #[inline]
                #[allow(unused_variables)]
                fn sub(self, rhs: Zero) -> Self::Output {
                    self
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn sum_zero_and_vector() {
        let algebra = Algebra::new(3, 0, 0);
        let zero = AlgebraType::Zero(algebra);
        let vector = AlgebraType::Grade(algebra.grade(1));

        let actual = &SumOp::Sub.item_impl(zero, vector)[0];
        let expected = parse_quote! {
            impl<T> std::ops::Sub<Vector<T>> for Zero
            where
                T: num_traits::Float
            {
                type Output = Vector<T>;
                #[inline]
                #[allow(unused_variables)]
                fn sub(self, rhs: Vector<T>) -> Self::Output {
                    -rhs
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn sum_vector_and_bivector() {
        let algebra = Algebra::new(3, 0, 0);
        let vector = AlgebraType::Grade(algebra.grade(1));
        let bivector = AlgebraType::Grade(algebra.grade(2));

        let actual = &SumOp::Add.item_impl(vector, bivector)[0];
        let expected = parse_quote! {
            impl<T> std::ops::Add<Bivector<T>> for Vector<T>
            where
                T: num_traits::Float
            {
                type Output = Multivector<Zero, Vector<T>, Bivector<T>, Zero>;
                #[inline]
                #[allow(unused_variables)]
                fn add(self, rhs: Bivector<T>) -> Self::Output {
                    Multivector(Zero, self, rhs, Zero)
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn sum_multivectors() {
        let algebra = Algebra::new(2, 0, 0);
        let mv = algebra.mv();

        let actual = &SumOp::Add.item_impl(mv, mv)[0];
        let expected = parse_quote! {
            impl<G0Lhs, G1Lhs, G2Lhs, G0Rhs, G1Rhs, G2Rhs, G0Out, G1Out, G2Out> std::ops::Add<Multivector<G0Rhs, G1Rhs, G2Rhs>>
            for Multivector<G0Lhs, G1Lhs, G2Lhs>
            where
                G0Lhs: GradeAdd<G0Rhs, Output = G0Out>,
                G1Lhs: GradeAdd<G1Rhs, Output = G1Out>,
                G2Lhs: GradeAdd<G2Rhs, Output = G2Out>
            {
                type Output = Multivector<G0Out, G1Out, G2Out>;
                #[inline]
                #[allow(unused_variables)]
                fn add(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs>) -> Self::Output {
                    Multivector(
                        GradeAdd::add(self.0, rhs.0),
                        GradeAdd::add(self.1, rhs.1),
                        GradeAdd::add(self.2, rhs.2)
                    )
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn sum_multivector_and_vector() {
        let algebra = Algebra::new(2, 0, 0);
        let mv = algebra.mv();
        let vector = AlgebraType::Grade(algebra.grade(1));

        let actual = &SumOp::Add.item_impl(mv, vector)[0];
        let expected = parse_quote! {
            impl<T, G0Lhs, G1Lhs, G2Lhs, G1Out> std::ops::Add<Vector<T>>
            for Multivector<G0Lhs, G1Lhs, G2Lhs>
            where
                T: num_traits::Float,
                G1Lhs: GradeAdd<Vector<T>, Output = G1Out>
            {
                type Output = Multivector<G0Lhs, G1Out, G2Lhs>;
                #[inline]
                #[allow(unused_variables)]
                fn add(self, rhs: Vector<T>) -> Self::Output {
                    Multivector(
                        self.0,
                        GradeAdd::add(self.1, rhs),
                        self.2
                    )
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn grade_add_sub_for_mv_is_none() {
        let algebra = Algebra::new(2, 0, 0);
        let mv = algebra.mv();
        let vector = AlgebraType::Grade(algebra.grade(1));

        assert!(SumOp::GradeAdd.item_impl(mv, vector).is_empty());
        assert!(SumOp::GradeSub.item_impl(mv, vector).is_empty());
        assert!(SumOp::GradeAdd.item_impl(vector, mv).is_empty());
        assert!(SumOp::GradeSub.item_impl(vector, mv).is_empty());
    }

    #[test]
    fn grade_add_vectors() {
        let algebra = Algebra::new(2, 0, 0);
        let vector = AlgebraType::Grade(algebra.grade(1));

        let actual = &SumOp::GradeAdd.item_impl(vector, vector)[0];

        let expected = parse_quote! {
            impl<T> GradeAdd<Vector<T>> for Vector<T>
            where
                T: num_traits::Float
            {
                type Output = Vector<T>;
                #[inline]
                #[allow(unused_variables)]
                fn add(self, rhs: Vector<T>) -> Self::Output {
                    Vector {
                        e1: std::ops::Add::add(self.e1, rhs.e1),
                        e2: std::ops::Add::add(self.e2, rhs.e2),
                    }
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn zero_sub_vector() {
        let algebra = Algebra::new(2, 0, 0);
        let zero = AlgebraType::Zero(algebra);
        let vector = AlgebraType::Grade(algebra.grade(1));

        let actual = &SumOp::Sub.item_impl(zero, vector)[0];

        let expected = parse_quote! {
            impl<T> std::ops::Sub<Vector<T>> for Zero
            where
                T: num_traits::Float
            {
                type Output = Vector<T>;
                #[inline]
                #[allow(unused_variables)]
                fn sub(self, rhs: Vector<T>) -> Self::Output {
                    -rhs
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn zero_grade_sub_vector() {
        let algebra = Algebra::new(2, 0, 0);
        let zero = AlgebraType::Zero(algebra);
        let vector = AlgebraType::Grade(algebra.grade(1));

        let actual = &SumOp::GradeSub.item_impl(zero, vector)[0];

        let expected = parse_quote! {
            impl<T> GradeSub<Vector<T>> for Zero
            where
                T: num_traits::Float
            {
                type Output = Vector<T>;
                #[inline]
                #[allow(unused_variables)]
                fn sub(self, rhs: Vector<T>) -> Self::Output {
                    -rhs
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn zero_sub_mv() {
        let algebra = Algebra::new(2, 0, 0);
        let zero = AlgebraType::Zero(algebra);
        let mv = algebra.mv();

        let actual = &SumOp::Sub.item_impl(zero, mv)[0];

        let expected = parse_quote! {
            impl<G0Rhs, G1Rhs, G2Rhs, G0Out, G1Out, G2Out> std::ops::Sub<Multivector<G0Rhs, G1Rhs, G2Rhs>> for Zero
            where
                G0Rhs: std::ops::Neg<Output = G0Out>,
                G1Rhs: std::ops::Neg<Output = G1Out>,
                G2Rhs: std::ops::Neg<Output = G2Out>
            {
                type Output = Multivector<G0Out, G1Out, G2Out>;
                #[inline]
                #[allow(unused_variables)]
                fn sub(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs>) -> Self::Output {
                    -rhs
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn rev_mv() {
        let algebra = Algebra::new(2, 0, 0);
        let mv = algebra.mv();

        let actual = &UnaryOp::Reverse.impl_item(mv)[0];

        let expected = parse_quote! {
            impl<G0, G1, G2, G0Out, G1Out, G2Out> Reverse for Multivector<G0, G1, G2>
            where
                G0: Reverse<Output = G0Out>,
                G1: Reverse<Output = G1Out>,
                G2: Reverse<Output = G2Out>
            {
                type Output = Multivector<G0Out, G1Out, G2Out>;
                #[inline]
                fn rev(self) -> Self::Output {
                    Multivector(
                        Reverse::rev(self.0),
                        Reverse::rev(self.1),
                        Reverse::rev(self.2)
                    )
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn comp_mv() {
        let algebra = Algebra::new(2, 0, 0);
        let mv = algebra.mv();

        let actual = &UnaryOp::RightComplement.impl_item(mv)[0];

        let expected = parse_quote! {
            impl<G0, G1, G2, G0Out, G1Out, G2Out> RightComplement for Multivector<G0, G1, G2>
            where
                G0: RightComplement<Output = G2Out>,
                G1: RightComplement<Output = G1Out>,
                G2: RightComplement<Output = G0Out>
            {
                type Output = Multivector<G0Out, G1Out, G2Out>;
                #[inline]
                fn right_comp(self) -> Self::Output {
                    Multivector(
                        RightComplement::right_comp(self.2),
                        RightComplement::right_comp(self.1),
                        RightComplement::right_comp(self.0)
                    )
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn scalar_sub_mv() {
        let algebra = Algebra::new(2, 0, 0);
        let scalar = AlgebraType::Grade(algebra.grade(0));
        let mv = algebra.mv();

        let actual = &SumOp::Sub.item_impl(scalar, mv)[0];

        let expected = parse_quote! {
            impl<G0Rhs, G1Rhs, G2Rhs, G0Out, G1Out, G2Out> std::ops::Sub<Multivector<G0Rhs, G1Rhs, G2Rhs>> for f32
            where
                f32: GradeSub<G0Rhs, Output = G0Out>,
                G1Rhs: std::ops::Neg<Output = G1Out>,
                G2Rhs: std::ops::Neg<Output = G2Out>
            {
                type Output = Multivector<G0Out, G1Out, G2Out>;
                #[inline]
                #[allow(unused_variables)]
                fn sub(self, rhs: Multivector<G0Rhs, G1Rhs, G2Rhs>) -> Self::Output {
                    Multivector(GradeSub::sub(self, rhs.0), -rhs.1, -rhs.2)
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn vector_new() {
        let algebra = Algebra::new(2, 0, 0);
        let vector = AlgebraType::Grade(algebra.grade(1));

        let actual = vector.new_fn().unwrap();
        let expected = parse_quote! {
            #[inline]
            pub const fn new(e1: T, e2: T) -> Vector<T> {
                Vector {
                    e1,
                    e2,
                }
            }
        };

        assert_eq_ref(&expected, &actual, "new fn");
    }

    #[test]
    fn unary_op_mv_output() {
        let algebra = Algebra::new(3, 0, 1);

        let mv = AlgebraType::Multivector({
            let mut mv = Multivector::new(algebra);
            mv.insert(algebra.grade(0));
            mv.insert(algebra.grade(1));
            mv
        });

        let actual = UnaryOp::RightComplement.output(mv);

        let expected = AlgebraType::Multivector({
            let mut mv = Multivector::new(algebra);
            mv.insert(algebra.grade(4));
            mv.insert(algebra.grade(3));
            mv
        });

        assert_eq!(expected, actual);
    }

    #[test]
    fn add_zero_and_f64() {
        let algebra = Algebra::new(3, 0, 0);
        let zero = AlgebraType::Zero(algebra);
        let scalar = AlgebraType::Grade(algebra.grade(0));

        let actual = SumOp::Add.item_impl(zero, scalar);
        assert_eq!(2, actual.len());

        let actual = &actual[0];

        println!("{}", actual.to_token_stream().to_string());

        let expected = parse_quote! {
            impl std::ops::Add<f32> for Zero {
                type Output = f32;
                #[inline]
                #[allow(unused_variables)]
                fn add(self, rhs: f32) -> Self::Output {
                    rhs
                }
            }
        };

        assert_eq_impl(&expected, actual)
    }

    #[test]
    fn add_f64_and_vector() {
        let algebra = Algebra::new(4, 1, 0);
        let scalar = AlgebraType::Grade(algebra.grade(0));
        let vector = AlgebraType::Grade(algebra.grade(1));

        let actual = SumOp::Add.item_impl(scalar, vector);
        assert_eq!(2, actual.len());

        let actual = &actual[0];
        let expected = parse_quote! {
            impl std::ops::Add<Vector<f32>> for f32 {
                type Output = Multivector<f32, Vector<f32>, Zero, Zero, Zero, Zero>;
                #[inline]
                #[allow(unused_variables)]
                fn add(self, rhs: Vector<f32>) -> Self::Output {
                    Multivector(self, rhs, Zero, Zero, Zero, Zero)
                }
            }
        };

        assert_eq_impl(&expected, actual)
    }

    #[test]
    fn impl_item_to_const() {
        let item_impl: ItemImpl = parse_quote! {
            #[attr]
            impl<T> Default for A<T> where T: Default {
                fn default() -> Self {
                    todo!()
                }
            }
        };

        let expected = quote! {
            #[attr]
            impl<T> const Default for A<T> where T: Default {
                fn default() -> Self {
                    todo!()
                }
            }
        };

        assert_eq!(expected.to_string(), impl_const(&item_impl).to_string());
    }

    #[test]
    fn add_vector_and_scalar_without_scalar_type() {
        let algebra = Algebra::new(4, 1, 0);
        let scalar = AlgebraType::Grade(algebra.grade(0));
        let vector = AlgebraType::Grade(algebra.grade(1));

        let actual = &SumOp::Add.item_impl(vector, scalar)[0];
        let expected = parse_quote! {
            impl<T> std::ops::Add<T> for Vector<T>
            where
                T: num_traits::Float
            {
                type Output = Multivector<T, Vector<T>, Zero, Zero, Zero, Zero>;
                #[inline]
                #[allow(unused_variables)]
                fn add(self, rhs: T) -> Self::Output {
                    Multivector(rhs, self, Zero, Zero, Zero, Zero)
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn mul_f64_and_vector() {
        let algebra = Algebra::new(3, 0, 0);
        let scalar = AlgebraType::Grade(algebra.grade(0));
        let vector = AlgebraType::Grade(algebra.grade(1));

        let actual = &ProductOp::Mul.item_impl(scalar, vector)[1];
        let expected = parse_quote! {
            impl std::ops::Mul<Vector<f64>> for f64 {
                type Output = Vector<f64>;
                #[inline]
                #[allow(unused_variables)]
                fn mul(self, rhs: Vector<f64>) -> Self::Output {
                    Vector {
                        e1: self * rhs.e1,
                        e2: self * rhs.e2,
                        e3: self * rhs.e3,
                    }
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    fn add_scalar_and_scalar_is_none() {
        let algebra = Algebra::new(3, 0, 0);
        let scalar = AlgebraType::Grade(algebra.grade(0));

        let actual = SumOp::Add.item_impl(scalar, scalar);
        assert!(actual.is_empty());
    }

    #[test]
    fn mul_zero_and_zero() {
        let algebra = Algebra::new(2, 0, 0);
        let zero = AlgebraType::Zero(algebra);
        let actual = ProductOp::Mul.item_impl(zero, zero);
        assert_eq!(1, actual.len());
        let actual = &actual[0];

        let expected = parse_quote! {
            impl std::ops::Mul<Zero> for Zero {
                type Output = Zero;
                #[inline]
                #[allow(unused_variables)]
                fn mul(self, rhs: Zero) -> Self::Output {
                    Zero
                }
            }
        };

        assert_eq_impl(&expected, actual);
    }

    #[test]
    #[ignore]
    fn write_algebra() {
        let algebra = Algebra::new(3, 0, 1);
        write_to_file(&algebra.define());
    }
}

use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use std::collections::{btree_map::Entry, BTreeMap};

use crate::unary::UnaryTrait;
use crate::FloatParam::*;
use crate::{blade::Blade, Algebra, Insert, TraitBounds, Type};

#[derive(Debug)]
pub struct Constructor<'a> {
    algebra: &'a Algebra,
    bounds: &'a mut TraitBounds,
    ty: Type,
    pub blades: BTreeMap<Blade, TokenStream>,
}

impl<'a> Constructor<'a> {
    pub fn new<I, F, B>(
        algebra: &'a Algebra,
        bounds: &'a mut TraitBounds,
        blades: I,
        mut f: F,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = B>,
        F: FnMut(B, &mut TraitBounds) -> Option<ConstructorItem>,
    {
        let blades = blades
            .into_iter()
            .filter_map(|blade| f(blade, bounds))
            .collect::<Vec<_>>();

        let blades = blades.into_iter().fold(BTreeMap::new(), |mut map, item| {
            let ConstructorItem { output, tokens, .. } = item;
            match map.entry(output.unsigned()) {
                Entry::Vacant(entry) => {
                    let value = if output.is_negative() {
                        bounds.insert(T.neg());
                        quote! { -#tokens }
                    } else {
                        quote! { #tokens }
                    };
                    entry.insert(value);
                }
                Entry::Occupied(mut entry) => {
                    let append = &if output.is_negative() {
                        bounds.insert(T.sub(T, T));
                        quote! { - #tokens }
                    } else {
                        bounds.insert(T.add(T, T));
                        quote! { + #tokens }
                    };
                    append.to_tokens(entry.get_mut())
                }
            }
            map
        });

        let ty: Type = blades.keys().copied().collect::<Option<Type>>()?;

        let mut constructor = Self {
            algebra,
            bounds,
            ty,
            blades,
        };
        constructor.check_for_invalid_blades();
        constructor.fill_missing_blades();
        Some(constructor)
    }

    pub fn unary<I, F, B>(
        algebra: &'a Algebra,
        bounds: &'a mut TraitBounds,
        blades: I,
        f: F,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = B>,
        F: FnMut(B) -> Option<ConstructorItem>,
    {
        let blades = blades
            .into_iter()
            .filter_map(f)
            .fold(BTreeMap::new(), |mut map, item| {
                let ConstructorItem { output, tokens, .. } = item;
                match map.entry(output.unsigned()) {
                    Entry::Vacant(entry) => {
                        let value = if output.is_negative() {
                            bounds.insert(T.neg());
                            quote! { -#tokens }
                        } else {
                            quote! { #tokens }
                        };
                        entry.insert(value);
                    }
                    Entry::Occupied(mut entry) => {
                        let append = &if output.is_negative() {
                            bounds.insert(T.sub(T, T));
                            quote! { - #tokens }
                        } else {
                            bounds.insert(T.add(T, T));
                            quote! { + #tokens }
                        };
                        append.to_tokens(entry.get_mut())
                    }
                }
                map
            });

        let ty: Type = blades.keys().copied().collect::<Option<Type>>()?;

        let mut constructor = Self {
            algebra,
            bounds,
            ty,
            blades,
        };
        constructor.check_for_invalid_blades();
        constructor.fill_missing_blades();
        Some(constructor)
    }

    pub fn binary<I, F, B>(
        algebra: &'a Algebra,
        bounds: &'a mut TraitBounds,
        blades: I,
        f: F,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = B>,
        F: FnMut(B) -> Option<ConstructorItem>,
    {
        let blades = blades
            .into_iter()
            .filter_map(f)
            .fold(BTreeMap::new(), |mut map, item| {
                let ConstructorItem { output, tokens, .. } = item;
                match map.entry(output.unsigned()) {
                    Entry::Vacant(entry) => {
                        let value = if output.is_negative() {
                            bounds.insert(V.neg());
                            quote! { -#tokens }
                        } else {
                            quote! { #tokens }
                        };
                        entry.insert(value);
                    }
                    Entry::Occupied(mut entry) => {
                        let append = &if output.is_negative() {
                            bounds.insert(V.sub(V, V));
                            quote! { - #tokens }
                        } else {
                            bounds.insert(V.add(V, V));
                            quote! { + #tokens }
                        };
                        append.to_tokens(entry.get_mut())
                    }
                }
                map
            });

        let ty: Type = blades.keys().copied().collect::<Option<Type>>()?;

        let mut constructor = Self {
            algebra,
            bounds,
            ty,
            blades,
        };
        constructor.check_for_invalid_blades();
        constructor.fill_missing_blades();
        Some(constructor)
    }

    pub fn set_type(&mut self, ty: Type) {
        self.ty = ty;
        self.check_for_invalid_blades();
        self.fill_missing_blades();
    }

    fn fill_missing_blades(&mut self) {
        let Self {
            algebra,
            bounds,
            ty,
            blades,
        } = self;

        let (zero_ty, zero_fn) = UnaryTrait::Zero.ty_fn();

        for blade in algebra.type_blades(*ty) {
            if let Entry::Vacant(entry) = blades.entry(blade) {
                bounds.insert(V.zero());
                entry.insert(quote! { #zero_ty::#zero_fn() });
            }
        }
    }

    fn check_for_invalid_blades(&self) {
        let expected_blades = self
            .algebra
            .type_blades(self.ty)
            .collect::<std::collections::HashSet<_>>();
        assert!(
            self.blades
                .keys()
                .all(|blade| expected_blades.contains(blade)),
            "invalid blades for given type"
        );
    }

    pub fn ty(&self) -> Type {
        self.ty
    }

    pub fn into_tokens(self) -> TokenStream {
        self.to_token_stream()
    }
}

impl<'a> ToTokens for Constructor<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let ty = self.ty;

        let fields = self.blades.iter().map(|(blade, tokens)| {
            let field = &self.algebra.fields[*blade];
            quote! {
                #field: #tokens,
            }
        });

        tokens.extend(quote! {
            #ty {
                #(#fields)*
                marker: std::marker::PhantomData,
            }
        });
    }
}

pub struct ConstructorItem {
    output: Blade,
    tokens: TokenStream,
}

impl ConstructorItem {
    pub fn new(output: Blade, tokens: TokenStream) -> Option<Self> {
        assert!(!tokens.is_empty());

        if output.is_zero() {
            return None;
        }

        Some(Self { output, tokens })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Basis, Insert, TraitBounds};
    use itertools::iproduct;

    fn ga_3d() -> Algebra {
        Algebra::new([Basis::pos('x'), Basis::pos('y'), Basis::pos('z')])
    }

    #[test]
    fn bivector_reverse() {
        let algebra = ga_3d();
        let bivector = Type::Grade(2);
        let mut bounds = TraitBounds::default();

        let constructor = Constructor::unary(
            &algebra,
            &mut bounds,
            algebra.type_blades(bivector),
            |input| {
                let output = input.rev();
                let field = &algebra.fields[input];
                let tokens = quote! { self.#field };
                ConstructorItem::new(output, tokens)
            },
        )
        .unwrap();

        assert_eq! {
            quote! {
                Bivector {
                    xy: -self.xy,
                    xz: -self.xz,
                    yz: -self.yz,
                    marker: std::marker::PhantomData,
                }
            }.to_string(),
            constructor.to_token_stream().to_string()
        }
    }

    #[test]
    fn bivector_dual() {
        let algebra = ga_3d();
        let bivector = Type::Grade(2);
        let mut bounds = TraitBounds::default();
        bounds.insert(T);

        let constructor = Constructor::unary(
            &algebra,
            &mut bounds,
            algebra.type_blades(bivector),
            |input| {
                let output = algebra.left_comp(input);
                let field = &algebra.fields[input];
                let tokens = quote! { self.#field };
                ConstructorItem::new(output, tokens)
            },
        )
        .unwrap();

        assert_eq! {
            quote! {
                Vector {
                    x: self.yz,
                    y: -self.xz,
                    z: self.xy,
                    marker: std::marker::PhantomData,
                }
            }.to_string(),
            constructor.to_token_stream().to_string()
        }

        let (params, where_clause) = bounds.params_and_where_clause();
        assert_eq!(quote! { <T> }.to_string(), params.to_string());
        assert_eq!(
            quote! {
               where
                   T: std::ops::Neg<Output = T>,
            }
            .to_string(),
            where_clause.to_string()
        )
    }

    #[test]
    fn vector_sum() {
        let algebra = ga_3d();
        let vector = Type::Grade(1);

        let mut bounds = TraitBounds::default();
        bounds.insert(T.add(U, V));

        let constructor = Constructor::binary(
            &algebra,
            &mut bounds,
            algebra.type_fields(vector),
            |(blade, field)| ConstructorItem::new(blade, quote! { self.#field + rhs.#field }),
        );

        assert_eq! {
            quote! {
                Vector {
                    x: self.x + rhs.x,
                    y: self.y + rhs.y,
                    z: self.z + rhs.z,
                    marker: std::marker::PhantomData,
                }
            }.to_string(),
            constructor.to_token_stream().to_string(),
        }

        let (params, where_clause) = bounds.params_and_where_clause();
        assert_eq!(quote! { <T, U, V> }.to_string(), params.to_string());
        assert_eq!(
            quote! {
               where
                   T: std::ops::Add<U, Output = V>,
            }
            .to_string(),
            where_clause.to_string()
        )
    }

    #[test]
    fn vector_product() {
        let algebra = ga_3d();
        let vector = Type::Grade(1);

        let mut bounds = TraitBounds::default();
        bounds.insert(T.copy());
        bounds.insert(U.copy());
        bounds.insert(T.mul(U, V));

        let blades = algebra.type_fields(vector);
        let iter = iproduct!(blades.clone(), blades);
        let constructor =
            Constructor::binary(&algebra, &mut bounds, iter, |((lb, lf), (rb, rf))| {
                let output = algebra.geo(lb, rb);
                ConstructorItem::new(output, quote! { self.#lf * rhs.#rf })
            });

        assert_eq! {
            quote! {
                Motor {
                    s: self.x * rhs.x + self.y * rhs.y + self.z * rhs.z,
                    xy: self.x * rhs.y - self.y * rhs.x,
                    xz: self.x * rhs.z - self.z * rhs.x,
                    yz: self.y * rhs.z - self.z * rhs.y,
                    marker: std::marker::PhantomData,
                }
            }
            .to_string(),
            constructor.to_token_stream().to_string(),
        };

        let (params, where_clause) = bounds.params_and_where_clause();
        assert_eq!(quote! { <T, U, V> }.to_string(), params.to_string());
        assert_eq!(
            quote! {
               where
                   T: std::marker::Copy + std::ops::Mul<U, Output = V>,
                   U: std::marker::Copy,
                   V: std::ops::Add<V, Output = V> + std::ops::Sub<V, Output = V>,
            }
            .to_string(),
            where_clause.to_string()
        );
    }

    #[test]
    fn motor_one() {
        let algebra = ga_3d();
        let mut bounds = TraitBounds::default();
        let (one_ty, one_fn) = UnaryTrait::One.ty_fn();
        let (zero_ty, zero_fn) = UnaryTrait::Zero.ty_fn();
        let constructor = Constructor::new(
            &algebra,
            &mut bounds,
            algebra.type_blades(Type::Motor),
            |blade, bounds| {
                ConstructorItem::new(
                    blade,
                    if blade.is_scalar() {
                        bounds.insert(T.one());
                        quote! { #one_ty::#one_fn() }
                    } else {
                        bounds.insert(T.zero());
                        quote! { #zero_ty::#zero_fn() }
                    },
                )
            },
        )
        .unwrap();

        assert_eq! {
            quote! {
                Motor {
                    s: #one_ty::#one_fn(),
                    xy: #zero_ty::#zero_fn(),
                    xz: #zero_ty::#zero_fn(),
                    yz: #zero_ty::#zero_fn(),
                    marker: std::marker::PhantomData,
                }
            }.to_string(),
            constructor.to_token_stream().to_string()
        }
    }
}

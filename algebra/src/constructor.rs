use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use std::collections::{btree_map::Entry, BTreeMap};

use crate::FloatParam::*;
use crate::{blade::Blade, Algebra, Insert, TraitBounds, Type};

#[derive(Debug)]
pub struct Constructor<'a> {
    algebra: &'a Algebra,
    pub ty: Type,
    blades: BTreeMap<Blade, TokenStream>,
}

impl<'a> Constructor<'a> {
    pub fn new<I, F, B>(algebra: &'a Algebra, blades: I, f: F) -> Option<Self>
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
                            quote! { -#tokens }
                        } else {
                            quote! { #tokens }
                        };
                        entry.insert(value);
                    }
                    Entry::Occupied(mut entry) => {
                        let append = &if output.is_negative() {
                            quote! { - #tokens }
                        } else {
                            quote! { + #tokens }
                        };
                        append.to_tokens(entry.get_mut())
                    }
                }
                map
            });
        let ty = blades.keys().copied().collect::<Option<Type>>()?;
        Some(Self {
            algebra,
            ty,
            blades,
        })
    }

    pub fn unary<I, F, B>(
        algebra: &'a Algebra,
        bounds: &mut TraitBounds,
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
        Some(Self {
            algebra,
            ty,
            blades,
        })
    }

    pub fn binary<I, F, B>(
        algebra: &'a Algebra,
        bounds: &mut TraitBounds,
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
        Some(Self {
            algebra,
            ty,
            blades,
        })
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
    use crate::{Basis, FloatParam, Insert, TraitBounds, TypeBlades, TypeFields};
    use itertools::iproduct;

    #[test]
    fn bivector_reverse() {
        let algebra = Algebra::new([Basis::pos('x'), Basis::pos('y'), Basis::pos('z')]);
        let bivector = Type::Grade(2);
        let mut bounds = TraitBounds::default();

        let constructor =
            Constructor::new(&algebra, TypeBlades::new(&algebra, bivector), |input| {
                let output = input.rev();
                let field = &algebra.fields[input];
                if output.is_negative() {
                    bounds.insert(FloatParam::T.neg());
                }
                let tokens = quote! { self.#field };
                Some(ConstructorItem { output, tokens })
            })
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
        let algebra = Algebra::new([Basis::pos('x'), Basis::pos('y'), Basis::pos('z')]);
        let bivector = Type::Grade(2);
        let mut bounds = TraitBounds::default();
        bounds.insert(T);

        let constructor = Constructor::unary(
            &algebra,
            &mut bounds,
            TypeBlades::new(&algebra, bivector),
            |input| {
                let output = algebra.left_comp(input);
                let field = &algebra.fields[input];
                let tokens = quote! { self.#field };
                Some(ConstructorItem { output, tokens })
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
        let algebra = Algebra::new([Basis::pos('x'), Basis::pos('y'), Basis::pos('z')]);
        let vector = Type::Grade(1);

        let mut bounds = TraitBounds::default();
        bounds.insert(T.add(U, V));

        let constructor = Constructor::binary(
            &algebra,
            &mut bounds,
            TypeFields::new(&algebra, vector),
            |(blade, field)| {
                Some(ConstructorItem {
                    output: blade,
                    tokens: quote! { self.#field + rhs.#field },
                })
            },
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
        let algebra = Algebra::new([Basis::pos('x'), Basis::pos('y'), Basis::pos('z')]);
        let vector = Type::Grade(1);

        let mut bounds = TraitBounds::default();
        bounds.insert(T.copy());
        bounds.insert(U.copy());
        bounds.insert(T.mul(U, V));

        let blades = TypeFields::new(&algebra, vector);
        let iter = iproduct!(blades.clone(), blades);
        let constructor =
            Constructor::binary(&algebra, &mut bounds, iter, |((lb, lf), (rb, rf))| {
                let output = algebra.geo(lb, rb);
                Some(ConstructorItem {
                    output,
                    tokens: quote! { self.#lf * rhs.#rf },
                })
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
}

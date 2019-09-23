#pragma once

#include <tuple>

namespace neural
{

namespace utils
{

template <typename InputLayer, typename... HiddenLayers>
constexpr int __get_size()
{
    if constexpr(sizeof...(HiddenLayers) == 0) return InputLayer::Size;
    else return InputLayer::Size * __get_size<HiddenLayers...>();
}

template <typename InputLayer, typename... HiddenLayers>
constexpr auto __get_info()
{
    if constexpr(sizeof...(HiddenLayers) == 0)
        return std::make_tuple(InputLayer::info);
    else return std::tuple_cat(std::make_tuple(InputLayer::info),
                               __get_info<HiddenLayers...>());
}

template <int n, typename... Ts>
const bool __ne(const std::tuple<Ts...> &lhs, const std::tuple<Ts...> &rhs)
{
    if constexpr(sizeof...(Ts) == n)
        return false;
    else return std::get<n>(lhs) != std::get<n>(rhs)
             || __ne<n+1, Ts...>(lhs, rhs);
}

template <typename... Ts>
const bool __ne(const std::tuple<Ts...> &lhs, const std::tuple<Ts...> &rhs)
{
    return __ne<0, Ts...>(lhs, rhs);
}

}

}
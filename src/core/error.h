#pragma once

#include <format>
#include <stdexcept>
#include <string>
#include <utility>

namespace tie {

// Base of every error tie raises deliberately. Messages always name the offending
// value (path, tensor, expected vs actual) so a failure is actionable on its own.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// A model file is malformed or inconsistent with what it claims to contain.
class LoadError : public Error {
 public:
  using Error::Error;
};

// Well-formed input that tie deliberately does not handle (yet).
class UnsupportedError : public Error {
 public:
  using Error::Error;
};

// A configured limit (KV blocks, context length, step size) would be exceeded.
class CapacityError : public Error {
 public:
  using Error::Error;
};

// A caller passed a value that can never be valid.
class InvalidArgument : public Error {
 public:
  using Error::Error;
};

template <typename E, typename... Args>
[[noreturn]] void fail(std::format_string<Args...> fmt, Args&&... args) {
  throw E(std::format(fmt, std::forward<Args>(args)...));
}

}  // namespace tie

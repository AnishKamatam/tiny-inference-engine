# Concatenates the Metal headers and kernels (in the given order) into one MSL
# source and writes a C++ file defining tie::kMetalSource. Local #include "..."
# lines and #pragma once are dropped because everything ends up in one unit.
#   -DOUTPUT=<file.cpp>  -DSOURCES=<files separated by |>
string(REPLACE "|" ";" source_files "${SOURCES}")
set(msl "")
foreach(file IN LISTS source_files)
  file(READ "${file}" text)
  string(REGEX REPLACE "#include \"[^\"]*\"[^\n]*\n" "" text "${text}")
  string(REPLACE "#pragma once" "" text "${text}")
  get_filename_component(file_name "${file}" NAME)
  string(APPEND msl "// ---- ${file_name}\n${text}\n")
endforeach()
file(WRITE "${OUTPUT}"
  "// Generated from src/kernels/metal by cmake/embed_metal.cmake. Do not edit.\n"
  "namespace tie {\n"
  "extern const char kMetalSource[];\n"
  "const char kMetalSource[] = R\"TIEMSL(${msl})TIEMSL\";\n"
  "}  // namespace tie\n")

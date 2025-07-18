/* Auto-generated by Tools/scripts/generate_opcode_h.py from Lib/opcode.py */
#ifndef Py_OPCODE_H
#define Py_OPCODE_H
#ifdef __cplusplus
extern "C" {
#endif


    /* Instruction opcodes for compiled code */
#define PY_OPCODES(X) \
  X(POP_TOP,                           1) \
  X(ROT_TWO,                           2) \
  X(ROT_THREE,                         3) \
  X(DUP_TOP,                           4) \
  X(DUP_TOP_TWO,                       5) \
  X(ROT_FOUR,                          6) \
  X(NOP,                               9) \
  X(UNARY_POSITIVE,                   10) \
  X(UNARY_NEGATIVE,                   11) \
  X(UNARY_NOT,                        12) \
  X(UNARY_INVERT,                     15) \
  X(BINARY_MATRIX_MULTIPLY,           16) \
  X(INPLACE_MATRIX_MULTIPLY,          17) \
  X(BINARY_POWER,                     19) \
  X(BINARY_MULTIPLY,                  20) \
  X(BINARY_MODULO,                    22) \
  X(BINARY_ADD,                       23) \
  X(BINARY_SUBTRACT,                  24) \
  X(BINARY_SUBSCR,                    25) \
  X(BINARY_FLOOR_DIVIDE,              26) \
  X(BINARY_TRUE_DIVIDE,               27) \
  X(INPLACE_FLOOR_DIVIDE,             28) \
  X(INPLACE_TRUE_DIVIDE,              29) \
  X(GET_LEN,                          30) \
  X(MATCH_MAPPING,                    31) \
  X(MATCH_SEQUENCE,                   32) \
  X(MATCH_KEYS,                       33) \
  X(COPY_DICT_WITHOUT_KEYS,           34) \
  X(WITH_EXCEPT_START,                49) \
  X(GET_AITER,                        50) \
  X(GET_ANEXT,                        51) \
  X(BEFORE_ASYNC_WITH,                52) \
  X(END_ASYNC_FOR,                    54) \
  X(INPLACE_ADD,                      55) \
  X(INPLACE_SUBTRACT,                 56) \
  X(INPLACE_MULTIPLY,                 57) \
  X(INPLACE_MODULO,                   59) \
  X(STORE_SUBSCR,                     60) \
  X(DELETE_SUBSCR,                    61) \
  X(BINARY_LSHIFT,                    62) \
  X(BINARY_RSHIFT,                    63) \
  X(BINARY_AND,                       64) \
  X(BINARY_XOR,                       65) \
  X(BINARY_OR,                        66) \
  X(INPLACE_POWER,                    67) \
  X(GET_ITER,                         68) \
  X(GET_YIELD_FROM_ITER,              69) \
  X(PRINT_EXPR,                       70) \
  X(LOAD_BUILD_CLASS,                 71) \
  X(YIELD_FROM,                       72) \
  X(GET_AWAITABLE,                    73) \
  X(LOAD_ASSERTION_ERROR,             74) \
  X(INPLACE_LSHIFT,                   75) \
  X(INPLACE_RSHIFT,                   76) \
  X(INPLACE_AND,                      77) \
  X(INPLACE_XOR,                      78) \
  X(INPLACE_OR,                       79) \
  X(LIST_TO_TUPLE,                    82) \
  X(RETURN_VALUE,                     83) \
  X(IMPORT_STAR,                      84) \
  X(SETUP_ANNOTATIONS,                85) \
  X(YIELD_VALUE,                      86) \
  X(POP_BLOCK,                        87) \
  X(POP_EXCEPT,                       89) \
  X(HAVE_ARGUMENT,                    90) \
  X(STORE_NAME,                       90) \
  X(DELETE_NAME,                      91) \
  X(UNPACK_SEQUENCE,                  92) \
  X(FOR_ITER,                         93) \
  X(UNPACK_EX,                        94) \
  X(STORE_ATTR,                       95) \
  X(DELETE_ATTR,                      96) \
  X(STORE_GLOBAL,                     97) \
  X(DELETE_GLOBAL,                    98) \
  X(ROT_N,                            99) \
  X(LOAD_CONST,                      100) \
  X(LOAD_NAME,                       101) \
  X(BUILD_TUPLE,                     102) \
  X(BUILD_LIST,                      103) \
  X(BUILD_SET,                       104) \
  X(BUILD_MAP,                       105) \
  X(LOAD_ATTR,                       106) \
  X(COMPARE_OP,                      107) \
  X(IMPORT_NAME,                     108) \
  X(IMPORT_FROM,                     109) \
  X(JUMP_FORWARD,                    110) \
  X(JUMP_IF_FALSE_OR_POP,            111) \
  X(JUMP_IF_TRUE_OR_POP,             112) \
  X(JUMP_ABSOLUTE,                   113) \
  X(POP_JUMP_IF_FALSE,               114) \
  X(POP_JUMP_IF_TRUE,                115) \
  X(LOAD_GLOBAL,                     116) \
  X(IS_OP,                           117) \
  X(CONTAINS_OP,                     118) \
  X(RERAISE,                         119) \
  X(JUMP_IF_NOT_EXC_MATCH,           121) \
  X(SETUP_FINALLY,                   122) \
  X(LOAD_FAST,                       124) \
  X(STORE_FAST,                      125) \
  X(DELETE_FAST,                     126) \
  X(GEN_START,                       129) \
  X(RAISE_VARARGS,                   130) \
  X(CALL_FUNCTION,                   131) \
  X(MAKE_FUNCTION,                   132) \
  X(BUILD_SLICE,                     133) \
  X(LOAD_CLOSURE,                    135) \
  X(LOAD_DEREF,                      136) \
  X(STORE_DEREF,                     137) \
  X(DELETE_DEREF,                    138) \
  X(CALL_FUNCTION_KW,                141) \
  X(CALL_FUNCTION_EX,                142) \
  X(SETUP_WITH,                      143) \
  X(EXTENDED_ARG,                    144) \
  X(LIST_APPEND,                     145) \
  X(SET_ADD,                         146) \
  X(MAP_ADD,                         147) \
  X(LOAD_CLASSDEREF,                 148) \
  X(MATCH_CLASS,                     152) \
  X(SETUP_ASYNC_WITH,                154) \
  X(FORMAT_VALUE,                    155) \
  X(BUILD_CONST_KEY_MAP,             156) \
  X(BUILD_STRING,                    157) \
  X(INVOKE_METHOD,                   158) \
  X(LOAD_FIELD,                      159) \
  X(LOAD_METHOD,                     160) \
  X(CALL_METHOD,                     161) \
  X(LIST_EXTEND,                     162) \
  X(SET_UPDATE,                      163) \
  X(DICT_MERGE,                      164) \
  X(DICT_UPDATE,                     165) \
  X(STORE_FIELD,                     166) \
  X(BUILD_CHECKED_LIST,              168) \
  X(LOAD_TYPE,                       169) \
  X(CAST,                            170) \
  X(LOAD_LOCAL,                      171) \
  X(STORE_LOCAL,                     172) \
  X(PRIMITIVE_BOX,                   174) \
  X(POP_JUMP_IF_ZERO,                175) \
  X(POP_JUMP_IF_NONZERO,             176) \
  X(PRIMITIVE_UNBOX,                 177) \
  X(PRIMITIVE_BINARY_OP,             178) \
  X(PRIMITIVE_UNARY_OP,              179) \
  X(PRIMITIVE_COMPARE_OP,            180) \
  X(LOAD_ITERABLE_ARG,               181) \
  X(LOAD_MAPPING_ARG,                182) \
  X(INVOKE_FUNCTION,                 183) \
  X(JUMP_IF_ZERO_OR_POP,             184) \
  X(JUMP_IF_NONZERO_OR_POP,          185) \
  X(FAST_LEN,                        186) \
  X(CONVERT_PRIMITIVE,               187) \
  X(INVOKE_NATIVE,                   189) \
  X(LOAD_CLASS,                      190) \
  X(BUILD_CHECKED_MAP,               191) \
  X(SEQUENCE_GET,                    192) \
  X(SEQUENCE_SET,                    193) \
  X(LIST_DEL,                        194) \
  X(REFINE_TYPE,                     195) \
  X(PRIMITIVE_LOAD_CONST,            196) \
  X(RETURN_PRIMITIVE,                197) \
  X(LOAD_METHOD_SUPER,               198) \
  X(LOAD_ATTR_SUPER,                 199) \
  X(TP_ALLOC,                        200) \
  X(LOAD_METHOD_STATIC,              203) \
  X(LOAD_METHOD_UNSHADOWED_METHOD,   205) \
  X(LOAD_METHOD_TYPE_METHODLIKE,     206) \
  X(BUILD_CHECKED_LIST_CACHED,       207) \
  X(TP_ALLOC_CACHED,                 208) \
  X(LOAD_ATTR_S_MODULE,              209) \
  X(LOAD_METHOD_S_MODULE,            210) \
  X(INVOKE_FUNCTION_CACHED,          211) \
  X(INVOKE_FUNCTION_INDIRECT_CACHED, 212) \
  X(BUILD_CHECKED_MAP_CACHED,        213) \
  X(LOAD_METHOD_STATIC_CACHED,       214) \
  X(PRIMITIVE_STORE_FAST,            215) \
  X(CAST_CACHED_OPTIONAL,            216) \
  X(CAST_CACHED,                     217) \
  X(CAST_CACHED_EXACT,               218) \
  X(CAST_CACHED_OPTIONAL_EXACT,      219) \
  X(LOAD_PRIMITIVE_FIELD,            220) \
  X(STORE_PRIMITIVE_FIELD,           221) \
  X(LOAD_OBJ_FIELD,                  222) \
  X(STORE_OBJ_FIELD,                 223) \
  X(INVOKE_METHOD_CACHED,            224) \
  X(BINARY_SUBSCR_TUPLE_CONST_INT,   225) \
  X(BINARY_SUBSCR_DICT_STR,          226) \
  X(BINARY_SUBSCR_LIST,              227) \
  X(BINARY_SUBSCR_TUPLE,             228) \
  X(BINARY_SUBSCR_DICT,              229) \
  X(LOAD_METHOD_UNCACHABLE,          230) \
  X(LOAD_METHOD_MODULE,              231) \
  X(LOAD_METHOD_TYPE,                232) \
  X(LOAD_METHOD_SPLIT_DICT_DESCR,    233) \
  X(LOAD_METHOD_SPLIT_DICT_METHOD,   234) \
  X(LOAD_METHOD_DICT_DESCR,          235) \
  X(LOAD_METHOD_DICT_METHOD,         236) \
  X(LOAD_METHOD_NO_DICT_METHOD,      237) \
  X(LOAD_METHOD_NO_DICT_DESCR,       238) \
  X(STORE_ATTR_SLOT,                 239) \
  X(STORE_ATTR_SPLIT_DICT,           240) \
  X(STORE_ATTR_DESCR,                241) \
  X(STORE_ATTR_UNCACHABLE,           242) \
  X(STORE_ATTR_DICT,                 243) \
  X(LOAD_ATTR_POLYMORPHIC,           244) \
  X(LOAD_ATTR_SLOT,                  245) \
  X(LOAD_ATTR_MODULE,                246) \
  X(LOAD_ATTR_TYPE,                  247) \
  X(LOAD_ATTR_SPLIT_DICT_DESCR,      248) \
  X(LOAD_ATTR_SPLIT_DICT,            249) \
  X(LOAD_ATTR_DICT_NO_DESCR,         250) \
  X(LOAD_ATTR_NO_DICT_DESCR,         251) \
  X(LOAD_ATTR_DICT_DESCR,            252) \
  X(LOAD_ATTR_UNCACHABLE,            253) \
  X(LOAD_GLOBAL_CACHED,              254) \
  X(SHADOW_NOP,                      255)

#ifdef NEED_OPCODE_JUMP_TABLES
static uint32_t _PyOpcode_RelativeJump[8] = {
    0U,
    0U,
    536870912U,
    67125248U,
    67141632U,
    0U,
    0U,
    0U,
};
static uint32_t _PyOpcode_Jump[8] = {
    0U,
    0U,
    536870912U,
    101695488U,
    67141632U,
    50429952U,
    0U,
    0U,
};
#endif /* OPCODE_TABLES */


enum {
#define OP(op, value) op = value,
PY_OPCODES(OP)
#undef OP
};

/* EXCEPT_HANDLER is a special, implicit block type which is created when
   entering an except handler. It is not an opcode but we define it here
   as we want it to be available to both frameobject.c and ceval.c, while
   remaining private.*/
#define EXCEPT_HANDLER 257

#define HAS_ARG(op) ((op) >= HAVE_ARGUMENT)

#ifdef __cplusplus
}
#endif
#endif /* !Py_OPCODE_H */

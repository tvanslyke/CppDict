#define BOOST_TEST_MODULE test hash_dict
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "dict_test.h"

BOOST_AUTO_TEST_SUITE( dict_test_suite )
#include "test_eq.h"
#include "test_equal_range.h"
#include "test_count.h"
#include "test_find.h"
#include "test_types.h"
BOOST_AUTO_TEST_SUITE_END()


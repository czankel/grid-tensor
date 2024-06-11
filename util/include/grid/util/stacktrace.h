
#include <execinfo.h>
#include <unistd.h>

inline void stacktrace()
{
  void *array[100];
  size_t size = backtrace(array, 10);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
}

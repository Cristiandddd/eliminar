#ifndef _MYEXCEPTIONS_H_
#define _MYEXCEPTIONS_H_

#include <exception>

class BadAlloc : public std::exception {
public:
    const char* what() const noexcept override {
        return "Memory allocation failed";
    }
};

#endif

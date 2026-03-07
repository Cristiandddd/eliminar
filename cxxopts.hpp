#ifndef CXXOPTS_HPP_INCLUDED
#define CXXOPTS_HPP_INCLUDED

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <sstream>

namespace cxxopts {
    
    template<typename T>
    class value {
    private:
        T* ptr;
    public:
        value(T& ref) : ptr(&ref) {}
        
        template<typename U>
        T as() const {
            return static_cast<T>(*ptr);
        }
    };
    
    class ParseResult {
    private:
        std::map<std::string, std::string> values;
    public:
        void set(const std::string& key, const std::string& val) {
            values[key] = val;
        }
        
        bool count(const std::string& key) const {
            return values.find(key) != values.end();
        }
        
        template<typename T>
        T as(const std::string& key) const {
            auto it = values.find(key);
            if (it != values.end()) {
                std::stringstream ss(it->second);
                T result;
                ss >> result;
                return result;
            }
            return T{};
        }
        
        template<typename T>
        cxxopts::value<T> operator[](const std::string& key) const {
            static T temp = as<T>(key);
            return cxxopts::value<T>(temp);
        }
    };
    
    class Options {
    private:
        std::string prog_name;
        std::string description;
        
    public:
        Options(const std::string& program, const std::string& desc) 
            : prog_name(program), description(desc) {}
        
        Options& positional_help(const std::string&) { return *this; }
        Options& show_positional_help() { return *this; }
        Options& set_width(int) { return *this; }
        Options& set_tab_expansion() { return *this; }
        Options& allow_unrecognised_options() { return *this; }
        Options& add_options() { return *this; }
        Options& parse_positional(const std::vector<std::string>&) { return *this; }
        
        // Simplified option parsing - just return empty result
        ParseResult parse(int argc, const char* argv[]) {
            ParseResult result;
            
            for (int i = 1; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "-h" || arg == "--help") {
                    result.set("help", "true");
                } else if (arg == "-v" || arg == "--verbose") {
                    result.set("verbose", "true");
                } else if ((arg == "-n" || arg == "--nodes") && i + 1 < argc) {
                    result.set("nodes", argv[++i]);
                } else if ((arg == "-o" || arg == "--memory") && i + 1 < argc) {
                    result.set("memory", argv[++i]);
                } else if ((arg == "-l" || arg == "--alpha") && i + 1 < argc) {
                    result.set("alpha", argv[++i]);
                } else if ((arg == "-c" || arg == "--strategies") && i + 1 < argc) {
                    result.set("strategies", argv[++i]);
                } else if ((arg == "-d" || arg == "--seed") && i + 1 < argc) {
                    result.set("seed", argv[++i]);
                } else if ((arg == "-e" || arg == "--naive") && i + 1 < argc) {
                    result.set("naive", argv[++i]);
                } else if ((arg == "-p" || arg == "--producers") && i + 1 < argc) {
                    result.set("producers", argv[++i]);
                } else if ((arg == "-t" || arg == "--teq") && i + 1 < argc) {
                    result.set("teq", argv[++i]);
                } else if ((arg == "-b" || arg == "--startingm") && i + 1 < argc) {
                    result.set("startingm", argv[++i]);
                } else if ((arg == "-i" || arg == "--initiala") && i + 1 < argc) {
                    result.set("initiala", argv[++i]);
                } else if (arg[0] != '-') {
                    result.set("fileoutput", arg);
                }
            }
            
            return result;
        }
        
        template<typename T>
        Options& operator()(const std::string&, const std::string&, cxxopts::value<T>) {
            return *this;
        }
    };
    
    namespace exceptions {
        class exception : public std::exception {
        public:
            const char* what() const noexcept override {
                return "cxxopts parsing error";
            }
        };
    }
}

#endif

#include <cstdlib>
#include <string>
#include <string_view>

#include <Eigen/Dense>

template <typename CharPointer>
bool StringRangeNotBlank(CharPointer a, CharPointer b) {
    int not_blank_count = 0;
    while (a < b) {
        const auto c = *a;
        if (!(c == ' ' || c == '\t' || c == '\r' || c == '\n')) {
            not_blank_count++;
        }
        a++;
    }
    return (not_blank_count > 0);
}

template <typename Function>
void LineParseNumbers(const std::string_view line, Function f) {
    int i = 0;
    char *tail;
    const char *ptr = line.data();
    const char *end_ptr = ptr + line.size();
    while (ptr < end_ptr) {
        double value = std::strtod(ptr, &tail);
        if (tail == ptr || tail > end_ptr) break;
        f(i++, value);
        ptr = tail;
    }
}

class StringLineIterator {
public:
    StringLineIterator(std::string_view text): text_(text) { }

    template <typename Function>
    void ForEachLine(Function f) {
        bool had_lines = false;
        auto line_begin = text_.data();
        auto end_ptr = line_begin + text_.size();
        int index = 0;
        for (auto ptr = text_.data(); true; ptr++) {
            if (*ptr == '\n' || ptr >= end_ptr) {
                if (StringRangeNotBlank(line_begin, ptr)) {
                    std::string_view line(line_begin, std::size_t(ptr - line_begin));
                    f(index++, line);
                    had_lines = true;
                } else {
                    if (had_lines) break;
                }
                line_begin = ptr + 1;
            }
            if (ptr >= end_ptr) break;
        }
    }
private:
    const std::string_view text_;
};

template <typename FunctionInit, typename FunctionElement>
void ParseTable(const std::string_view text, FunctionInit f_init, FunctionElement f_element) {
    StringLineIterator lines_iterator(text);
    int rows = 0, cols = 0;
    lines_iterator.ForEachLine([&](int i, std::string_view line) {
        if (i == 0) {
            LineParseNumbers(line, [&](int j, double v) { cols = j + 1; });
        }
        rows++;
    });
    f_init(rows, cols);
    lines_iterator.ForEachLine([&](int i, std::string_view line) {
        LineParseNumbers(line, [&](int j, double value) {
            f_element(i, j, value);
        });
    });
}


Eigen::MatrixXd ParseMatrix(const std::string_view text) {
    Eigen::MatrixXd m;
    ParseTable(text,
        [&](int rows, int cols) { m.resize(rows, cols); },
        [&](int i, int j, double value) { m(i, j) = value; }
    );
    return m;
}


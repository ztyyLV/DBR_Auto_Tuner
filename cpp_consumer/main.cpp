// Loads an auto-tuned CaptureVision template and runs it over a folder of
// images with the C++ edition of Dynamsoft Barcode Reader 11.x, so the recall
// and timings the tuner reported can be confirmed natively.
//
//   dbr_verify <template.json> <image folder> [template name] [-license KEY]
//
// The template file is the one emitted by `python -m autotune`; nothing in it
// is Python specific.

#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#endif

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "DynamsoftCaptureVisionRouter.h"

using namespace dynamsoft::license;
using namespace dynamsoft::cvr;
using namespace dynamsoft::dbr;
using namespace dynamsoft::basic_structures;

namespace fs = std::filesystem;

namespace {

// Public trial key from the DBR samples; override with -license.
const char* kDefaultLicense = "DLS2eyJoYW5kc2hha2VDb2RlIjoiMjAwMDAxLTEwNTI2NzQwMSJ9";

const char* kImageExtensions[] = {".jpg", ".jpeg", ".png", ".bmp", ".gif",
                                  ".tif", ".tiff", ".pdf", ".webp"};

bool IsImage(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(::tolower(c)); });
    for (const char* candidate : kImageExtensions) {
        if (ext == candidate) return true;
    }
    return false;
}

std::vector<fs::path> CollectImages(const fs::path& root) {
    std::vector<fs::path> files;
    if (fs::is_regular_file(root)) {
        if (IsImage(root)) files.push_back(root);
        return files;
    }
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && IsImage(entry.path())) files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

double Percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(q * (values.size() - 1) + 0.5);
    return values[std::min(index, values.size() - 1)];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "usage: dbr_verify <template.json> <image folder> "
                     "[template name] [-license KEY]\n";
        return 1;
    }

    const std::string templatePath = argv[1];
    const std::string imageRoot = argv[2];
    std::string templateName;
    std::string license = kDefaultLicense;

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "-license") == 0 && i + 1 < argc) {
            license = argv[++i];
        } else if (argv[i][0] != '-') {
            templateName = argv[i];
        }
    }

    char message[512] = {0};
    int code = CLicenseManager::InitLicense(license.c_str(), message, sizeof(message));
    if (code != ErrorCode::EC_OK && code != ErrorCode::EC_LICENSE_WARNING) {
        std::cout << "license initialisation failed (" << code << "): " << message << "\n";
        return 2;
    }

    CCaptureVisionRouter router;
    code = router.InitSettingsFromFile(templatePath.c_str(), message, sizeof(message));
    if (code != ErrorCode::EC_OK && code != ErrorCode::EC_UNSUPPORTED_JSON_KEY_WARNING) {
        std::cout << "template rejected (" << code << "): " << message << "\n";
        return 3;
    }
    if (code == ErrorCode::EC_UNSUPPORTED_JSON_KEY_WARNING) {
        std::cout << "warning: " << message << "\n";
    }

    if (templateName.empty() && router.GetParameterTemplateCount() > 0) {
        // Fall back to the first template the file defines.
        char nameBuffer[256] = {0};
        if (router.GetParameterTemplateName(0, nameBuffer, sizeof(nameBuffer)) == ErrorCode::EC_OK) {
            templateName = nameBuffer;
        }
    }

    const std::vector<fs::path> files = CollectImages(imageRoot);
    if (files.empty()) {
        std::cout << "no images found under " << imageRoot << "\n";
        return 4;
    }

    std::cout << "template : " << templateName << "  (" << templatePath << ")\n"
              << "images   : " << files.size() << " file(s) under " << imageRoot << "\n\n";

    std::vector<double> timings;
    int pagesWithCode = 0, barcodesTotal = 0;

    for (const auto& file : files) {
        const std::string path = file.string();
        const auto started = std::chrono::steady_clock::now();
        CCapturedResultArray* results = router.CaptureMultiPages(path.c_str(),
                                                                 templateName.c_str());
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started).count();

        int pageCount = results ? results->GetResultsCount() : 0;
        int found = 0;
        std::string first;
        std::string firstFormat;

        for (int i = 0; i < pageCount; ++i) {
            const CCapturedResult* result = results->GetResult(i);
            if (!result) continue;
            if (result->GetErrorCode() != ErrorCode::EC_OK &&
                result->GetErrorCode() != ErrorCode::EC_UNSUPPORTED_JSON_KEY_WARNING) {
                std::cout << "  " << file.filename().string() << " page " << (i + 1)
                          << " error " << result->GetErrorCode() << ": "
                          << result->GetErrorString() << "\n";
            }
            CDecodedBarcodesResult* barcodes = result->GetDecodedBarcodesResult();
            if (!barcodes) continue;
            const int items = barcodes->GetItemsCount();
            found += items;
            for (int j = 0; j < items; ++j) {
                const CBarcodeResultItem* item = barcodes->GetItem(j);
                if (item && first.empty()) {
                    first = item->GetText();
                    firstFormat = item->GetFormatString();
                }
            }
            barcodes->Release();
        }
        if (results) results->Release();

        const int pages = std::max(1, pageCount);
        barcodesTotal += found;
        if (found > 0) ++pagesWithCode;
        timings.push_back(ms / pages);

        std::cout << std::left << std::setw(34) << file.filename().string()
                  << std::right << std::setw(9) << std::fixed << std::setprecision(1) << ms
                  << " ms   " << std::setw(2) << found << " code(s)";
        if (!first.empty()) {
            if (first.size() > 44) first = first.substr(0, 41) + "...";
            std::cout << "   " << firstFormat << " | " << first;
        }
        std::cout << "\n";
    }

    double total = 0.0;
    for (double value : timings) total += value;

    std::cout << "\n--------------------------------------------------------------\n"
              << "pages with >=1 barcode : " << pagesWithCode << "/" << files.size()
              << "  (" << std::fixed << std::setprecision(1)
              << (100.0 * pagesWithCode / static_cast<double>(files.size())) << "%)\n"
              << "barcodes decoded       : " << barcodesTotal << "\n"
              << "mean per page          : " << std::setprecision(1)
              << (timings.empty() ? 0.0 : total / timings.size()) << " ms\n"
              << "p95 per page           : " << Percentile(timings, 0.95) << " ms\n"
              << "worst page             : " << Percentile(timings, 1.0) << " ms\n";
    return 0;
}

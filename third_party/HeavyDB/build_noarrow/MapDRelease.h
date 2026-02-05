/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file		release.h
 * @brief		Defines the release number string
 *
 */

// clang-format off
#ifndef RELEASE_H
#define RELEASE_H

#include <string>

static const int32_t MAPD_VERSION_MAJOR{9};
static const int32_t MAPD_VERSION_MINOR{0};
static const int32_t MAPD_VERSION_PATCH{0};
static const int32_t MAPD_VERSION{
  9 * 1000000 + 0 * 1000 + 0
};

static const std::string MAPD_VERSION_EXTRA{"dev"};
static const std::string MAPD_VERSION_RAW{"9.0.0dev"};
static const std::string MAPD_BUILD_DATE{"20260129"};
static const std::string MAPD_GIT_HASH{"nogit"};
static const std::string MAPD_EDITION{"os"};

static const std::string MAPD_RELEASE{"9.0.0dev-20260129-nogit"};

#endif  // RELEASE_H

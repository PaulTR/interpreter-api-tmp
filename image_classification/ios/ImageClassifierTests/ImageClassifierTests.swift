// Copyright 2024 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import XCTest
@testable import ImageClassifier
import TensorFlowLiteTaskVision

final class ImageClassifierTests: XCTestCase {

  static let efficientnetLite0 = Model.efficientnetLite0
  static let efficientnetLite2 = Model.efficientnetLite2

  static let scoreThreshold: Float = 0.01
  static let maxResult: Int = 3

  static let testImage = UIImage(
    named: "cup.png",
    in:Bundle(for: ImageClassifierTests.self),
    compatibleWith: nil)!

  static let efficientnetLite0Results = [
    ClassificationCategory(
      index: 504,
      score: 0.6224187,
      label: "coffee mug",
      displayName: nil),
    ClassificationCategory(
      index: 968,
      score: 0.13291624,
      label: "cup",
      displayName: nil),
    ClassificationCategory(
      index: 899,
      score: 0.055026982,
      label: "water jug",
      displayName: nil),
  ]

  static let efficientnetLite2Results = [
    ClassificationCategory(
      index: 504,
      score: 0.4353558,
      label: "coffee mug",
      displayName: nil),
    ClassificationCategory(
      index: 968,
      score: 0.16430414,
      label: "cup",
      displayName: nil),
    ClassificationCategory(
      index: 725,
      score: 0.017306592,
      label: "pitcher",
      displayName: nil),
  ]

  func imageClassifierWithModel(
    _ model: Model,
    scoreThreshold: Float,
    maxResult: Int
  ) throws -> ImageClassifierService {
    let imageClassifierService = ImageClassifierService(
      model: model,
      scoreThreshold: scoreThreshold,
      maxResult: maxResult)
    return imageClassifierService!
  }

  func assertImageClassifierResultHasOneHead(
    _ imageClassifierResult: ClassificationResult
  ) {
    XCTAssertEqual(imageClassifierResult.classifications.count, 1)
    XCTAssertEqual(imageClassifierResult.classifications[0].headIndex, 0)
  }

  func assertCategoriesAreEqual(
    category: ClassificationCategory,
    expectedCategory: ClassificationCategory,
    indexInCategoryList: Int
  ) {
    XCTAssertEqual(
      category.index,
      expectedCategory.index,
      String(
        format: """
              category[%d].index and expectedCategory[%d].index are not equal.
              """, indexInCategoryList))
    XCTAssertEqual(
      category.score,
      expectedCategory.score,
      accuracy: 1e-3,
      String(
        format: """
              category[%d].score and expectedCategory[%d].score are not equal.
              """, indexInCategoryList))
    XCTAssertEqual(
      category.label,
      expectedCategory.label,
      String(
        format: """
              category[%d].categoryName and expectedCategory[%d].categoryName are \
              not equal.
              """, indexInCategoryList))
    XCTAssertEqual(
      category.displayName,
      expectedCategory.displayName,
      String(
        format: """
              category[%d].displayName and expectedCategory[%d].displayName are \
              not equal.
              """, indexInCategoryList))
  }

  func assertEqualCategoryArrays(
    categoryArray: [ClassificationCategory],
    expectedCategoryArray: [ClassificationCategory]
  ) {
    XCTAssertEqual(
      categoryArray.count,
      expectedCategoryArray.count)

    for (index, (category, expectedCategory)) in zip(categoryArray, expectedCategoryArray)
      .enumerated()
    {
      assertCategoriesAreEqual(
        category: category,
        expectedCategory: expectedCategory,
        indexInCategoryList: index)
    }
  }

  func assertResultsForClassify(
    image: UIImage,
    using imageClassifier: ImageClassifierService,
    equals expectedCategories: [ClassificationCategory]
  ) throws {
    let imageClassifierResult =
    try XCTUnwrap(
      imageClassifier.classify(image: image)!.imageClassifierResults[0])
    print(imageClassifierResult)
    assertImageClassifierResultHasOneHead(imageClassifierResult)
    assertEqualCategoryArrays(
      categoryArray:
        imageClassifierResult.classifications[0].categories,
      expectedCategoryArray: expectedCategories)
  }

  func testClassifyWithEfficientnetLite0Succeeds() throws {
    let imageClassifier = try imageClassifierWithModel(
      ImageClassifierTests.efficientnetLite0,
      scoreThreshold: ImageClassifierTests.scoreThreshold,
      maxResult: ImageClassifierTests.maxResult)
    try assertResultsForClassify(
      image: ImageClassifierTests.testImage,
      using: imageClassifier,
      equals: ImageClassifierTests.efficientnetLite0Results)
  }

  func testClassifyWithEfficientnetLite2Succeeds() throws {

    let imageClassifier = try imageClassifierWithModel(
      ImageClassifierTests.efficientnetLite2,
      scoreThreshold: ImageClassifierTests.scoreThreshold,
      maxResult: ImageClassifierTests.maxResult)
    try assertResultsForClassify(
      image: ImageClassifierTests.testImage,
      using: imageClassifier,
      equals: ImageClassifierTests.efficientnetLite2Results)
  }
}

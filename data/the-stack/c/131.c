//
//  JAScanner.h
//  AFNetworking
//
//  Created by Shepherd on 2019/3/5.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT CGFloat visibleHeight;
FOUNDATION_EXPORT CGFloat visibleWidth;
FOUNDATION_EXPORT CGFloat animationDuration;

@class JAScanView;

typedef NS_ENUM(NSInteger,JAScannerState) {
    JAScannerStateNone,
    JAScannerStateScanning,
    JAScannerStateCompleted,
    JAScannerStateLeaveScene,
};

typedef NS_ENUM(NSInteger,JAScannerOption) {
    JAScannerOptionQR = 1 << 0
};

@interface JAScanner : NSObject

@property (nonatomic,strong) UIImage *areaImage;
@property (nonatomic,strong) UIImage *scanningImage;
@property (nonatomic,copy) void (^animateBlock)(void);
/** Custom add some views */
@property (nonatomic,strong,readonly) JAScanView *containView;

/** flashlight status */
@property (nonatomic,assign,getter=isBulb) BOOL bulb;
@property (nonatomic,assign,getter=isHideBulb) BOOL hideBulb;

/** Default option is qr */
@property (nonatomic,assign) JAScannerOption option;
/** Default status is none */
@property (nonatomic,assign) JAScannerState state;

/**
 Create a scanner object
 
 Default size is {200,200} and centered
 
 @param view Camera occupy view
 @return scanner object
 */
+ (instancetype)scanAtView:(UIView *)view;

/**
 Create a scanner object
 
 @param view Camera occupy view
 @param frame Scan active area
 @return scanner object
 */
+ (instancetype)scanAtView:(UIView *)view
               visibleArea:(CGRect)frame;

/**
 Start recognize with handler

 @param completionHandler An handler which recognize action completed
 */
- (void)startWithCompletionHandler:(void (^)(NSString *result))completionHandler;

/**
 Stop recognize
 */
- (void)stopRecognize;

@end

NS_ASSUME_NONNULL_END

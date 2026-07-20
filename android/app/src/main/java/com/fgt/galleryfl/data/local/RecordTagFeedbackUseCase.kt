package com.fgt.galleryfl.data.local

class RecordTagFeedbackUseCase(
    private val feedbackStore: FeedbackStore,
    private val feedbackDao: FeedbackDao
) {
    suspend fun recordConfirmed(imageId: Long, classIndex: Int) {
        feedbackStore.recordConfirmed(classIndex)
        feedbackStore.recordDemand(classIndex)
        feedbackDao.insertFeedback(FeedbackEntity(imageId, classIndex, true))
    }

    suspend fun recordRejected(imageId: Long, classIndex: Int) {
        feedbackStore.recordRejected(classIndex)
        feedbackDao.insertFeedback(FeedbackEntity(imageId, classIndex, false))
    }

    /** Store an explicit corrected parent as a trusted categorical target. */
    suspend fun recordCorrection(imageId: Long, predictedClassIndex: Int, correctClassIndex: Int) {
        if (predictedClassIndex != correctClassIndex) {
            feedbackStore.recordRejected(predictedClassIndex)
        }
        feedbackStore.recordConfirmed(correctClassIndex)
        feedbackStore.recordDemand(correctClassIndex)
        feedbackDao.insertFeedback(FeedbackEntity(imageId, correctClassIndex, true))
    }
    
    suspend fun recordPredicted(classIndex: Int) {
        feedbackStore.recordPredicted(classIndex)
        feedbackStore.recordDemand(classIndex)
    }
}

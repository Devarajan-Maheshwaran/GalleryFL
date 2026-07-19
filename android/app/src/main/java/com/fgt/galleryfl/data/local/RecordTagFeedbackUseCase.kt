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
    
    suspend fun recordPredicted(classIndex: Int) {
        feedbackStore.recordPredicted(classIndex)
        feedbackStore.recordDemand(classIndex)
    }
}

package com.zhou.craft;

import com.zhou.craft.model.CraftOrder;
import com.zhou.craft.model.PaymentRecord;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class OrderLifecycleTests {

    private CraftOrder newOrder() {
        return new CraftOrder(1L, 1L, 1L, 2L,
                "测试定制作品", "测试用户", "测试创作者", 299.0, 10);
    }

    @Test
    void normalLifecycleShouldReachCompleted() {
        CraftOrder order = newOrder();
        assertEquals(CraftOrder.PENDING_PAYMENT, order.getStatus());

        order.markPaid();
        assertEquals(CraftOrder.PENDING_PRODUCTION, order.getStatus());

        order.updateProgress(30, "开始制作");
        assertEquals(CraftOrder.IN_PRODUCTION, order.getStatus());

        order.updateProgress(100, "制作完成");
        assertEquals(CraftOrder.PENDING_SHIPMENT, order.getStatus());

        order.markShipped();
        assertEquals(CraftOrder.SHIPPED, order.getStatus());

        order.markInTransit();
        assertEquals(CraftOrder.IN_TRANSIT, order.getStatus());

        order.markDelivered();
        assertEquals(CraftOrder.PENDING_ACCEPTANCE, order.getStatus());

        order.complete();
        assertEquals(CraftOrder.COMPLETED, order.getStatus());
        assertEquals(100, order.getProgress());
        assertNotNull(order.getCompletedAt());
    }

    @Test
    void paidCancellationShouldRequireRefundBeforeClosing() {
        CraftOrder order = newOrder();
        order.markPaid();
        order.updateProgress(40, "制作进行中");

        PaymentRecord payment = new PaymentRecord(
                1L, "PAY-TEST-001", "MOCK_WECHAT", order.getAmount());

        order.requestCancel();
        assertEquals(CraftOrder.CANCEL_REQUESTED, order.getStatus());

        payment.refund("用户申请取消");
        order.markCancelledAfterRefund();

        assertEquals("已退款", payment.getStatus());
        assertEquals("用户申请取消", payment.getRefundReason());
        assertNotNull(payment.getRefundedAt());
        assertEquals(CraftOrder.CANCELLED, order.getStatus());
    }

    @Test
    void invalidStateTransitionShouldBeRejected() {
        CraftOrder order = newOrder();

        assertThrows(IllegalStateException.class, order::markShipped);
        assertThrows(IllegalStateException.class, order::complete);

        order.markPaid();
        order.updateProgress(100, "制作完成");
        order.markShipped();

        assertThrows(IllegalStateException.class, order::markPaid);
    }
}

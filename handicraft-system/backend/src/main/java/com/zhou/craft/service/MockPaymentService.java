package com.zhou.craft.service;
import com.zhou.craft.model.*;import com.zhou.craft.repo.*;import org.springframework.stereotype.Service;import org.springframework.transaction.annotation.Transactional;import java.util.*;import java.util.Optional;
@Service public class MockPaymentService implements PaymentService{
private final PaymentRecordRepository payments;private final CraftOrderRepository orders;public MockPaymentService(PaymentRecordRepository p,CraftOrderRepository o){payments=p;orders=o;}
@Transactional public PaymentRecord pay(CraftOrder order,String provider){var old=payments.findByOrderId(order.getId());if(old.isPresent())return old.get();order.markPaid();orders.save(order);String p=provider==null||provider.isBlank()?"MOCK_WECHAT":provider;return payments.save(new PaymentRecord(order.getId(),"PAY"+System.currentTimeMillis()+UUID.randomUUID().toString().substring(0,6).toUpperCase(),p,order.getAmount()));}
@Transactional public PaymentRecord refund(CraftOrder order,String reason){var record=payments.findByOrderId(order.getId()).orElseThrow(()->new IllegalStateException("订单没有可退款的支付记录"));record.refund(reason==null||reason.isBlank()?"用户取消定制订单":reason);order.markCancelledAfterRefund();orders.save(order);return payments.save(record);}
public Optional<PaymentRecord> findByOrderId(Long orderId){return payments.findByOrderId(orderId);}
}
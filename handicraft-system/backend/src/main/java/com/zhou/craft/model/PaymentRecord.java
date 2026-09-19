package com.zhou.craft.model;
import jakarta.persistence.*;import java.time.LocalDateTime;
@Entity @Table(name="payment_record",uniqueConstraints=@UniqueConstraint(columnNames="orderId"))
public class PaymentRecord{
@Id @GeneratedValue(strategy=GenerationType.IDENTITY)private Long id;private Long orderId;@Column(unique=true)private String paymentNo;private String provider;private double amount;private String status;private String refundReason;private LocalDateTime createdAt;private LocalDateTime paidAt;private LocalDateTime refundedAt;
public PaymentRecord(){}public PaymentRecord(Long orderId,String paymentNo,String provider,double amount){this.orderId=orderId;this.paymentNo=paymentNo;this.provider=provider;this.amount=amount;this.status="已支付";this.createdAt=LocalDateTime.now();this.paidAt=createdAt;}
public Long getId(){return id;}public Long getOrderId(){return orderId;}public String getPaymentNo(){return paymentNo;}public String getProvider(){return provider;}public double getAmount(){return amount;}public String getStatus(){return status;}public String getRefundReason(){return refundReason;}public LocalDateTime getCreatedAt(){return createdAt;}public LocalDateTime getPaidAt(){return paidAt;}public LocalDateTime getRefundedAt(){return refundedAt;}
public void refund(String reason){if(!"已支付".equals(status))throw new IllegalStateException("当前支付记录不能退款");status="已退款";refundReason=reason;refundedAt=LocalDateTime.now();}
}
package com.zhou.craft.model;
import jakarta.persistence.*;import java.time.LocalDateTime;
@Entity @Table(name="shipment",uniqueConstraints=@UniqueConstraint(columnNames="orderId"))
public class Shipment{
@Id @GeneratedValue(strategy=GenerationType.IDENTITY)private Long id;private Long orderId;private String carrier;@Column(unique=true)private String trackingNo;private String status;private LocalDateTime shippedAt;private LocalDateTime deliveredAt;
public Shipment(){}public Shipment(Long orderId,String carrier,String trackingNo){this.orderId=orderId;this.carrier=carrier;this.trackingNo=trackingNo;this.status="已发货";this.shippedAt=LocalDateTime.now();}
public Long getId(){return id;}public Long getOrderId(){return orderId;}public String getCarrier(){return carrier;}public String getTrackingNo(){return trackingNo;}public String getStatus(){return status;}public LocalDateTime getShippedAt(){return shippedAt;}public LocalDateTime getDeliveredAt(){return deliveredAt;}
public void inTransit(){status="运输中";}public void delivered(){status="已签收";deliveredAt=LocalDateTime.now();}
}
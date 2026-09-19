package com.zhou.craft.model;
import jakarta.persistence.*;import java.time.LocalDateTime;
@Entity @Table(name="logistics_trace")
public class LogisticsTrace{
@Id @GeneratedValue(strategy=GenerationType.IDENTITY)private Long id;private Long shipmentId;private String status;private String location;@Column(length=600)private String description;private LocalDateTime occurredAt;
public LogisticsTrace(){}public LogisticsTrace(Long shipmentId,String status,String location,String description){this.shipmentId=shipmentId;this.status=status;this.location=location;this.description=description;this.occurredAt=LocalDateTime.now();}
public Long getId(){return id;}public Long getShipmentId(){return shipmentId;}public String getStatus(){return status;}public String getLocation(){return location;}public String getDescription(){return description;}public LocalDateTime getOccurredAt(){return occurredAt;}
}
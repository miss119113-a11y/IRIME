package com.zhou.craft.model;
import jakarta.persistence.*;import java.time.LocalDateTime;
@Entity @Table(name="work_favorite",uniqueConstraints=@UniqueConstraint(columnNames={"userId","workId"}))
public class WorkFavorite{
@Id @GeneratedValue(strategy=GenerationType.IDENTITY) private Long id;private Long userId;private Long workId;private LocalDateTime createdAt;
public WorkFavorite(){} public WorkFavorite(Long userId,Long workId){this.userId=userId;this.workId=workId;} @PrePersist void init(){createdAt=LocalDateTime.now();}
public Long getId(){return id;}public Long getUserId(){return userId;}public Long getWorkId(){return workId;}public LocalDateTime getCreatedAt(){return createdAt;}
}
import cadquery as cq

result = (
cq.Workplane("XY")
.circle(5)
.extrude(10)
.faces(">Z")
.workplane()
.rect(6, 6, forConstruction=True)
.vertices()
.circle(1)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')
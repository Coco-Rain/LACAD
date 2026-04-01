import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 20, 5)
.union(cq.Workplane().circle(8).extrude(10))
.union(cq.Workplane().sphere(6))
.compounds()
.faces(">Z")
.workplane()
.circle(3)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')